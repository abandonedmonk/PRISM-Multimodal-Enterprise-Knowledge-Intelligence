import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

ENTITY_EXTRACTION_PROMPT = """Extract entities and relations from each text chunk.
Return ONLY valid JSON matching without any escape characters or markdown formatting, and nothing else. The JSON should follow this schema:
{
  "results": [
    {
      "chunk_id": str,
      "entities": [
        {"name": str, "type": str, "description": str}
      ],
      "relations": [
        {"source": str, "target": str, "relation": str, "description": str, "strength": float}
      ]
    }
  ]
}

Rules:
- No markdown
- No extra text outside the JSON
- No backslash n or other escaped characters in the output (\\n, \\t, etc. are NOT needed)
- No extra top-level keys
- Return one result item for every input chunk_id
- Keep output concise; max 6 entities and max 6 relations per chunk
- strength in [0,1]
- Output must start with { and end with }
- For each chunk, extract 3-6 entities and 2-6 relations when possible
- Relation descriptions must explain why source relates to target, 10-24 words
- Do not output empty relations unless no explicit or implied relation exists
- Use canonical entity types only: Company, Person, FinancialMetric, Regulation, Product, Geography, Date, Industry, Concept
- Use canonical relation labels only: related_to, depends_on, part_of, regulates, owns, mentions, acquires, causes, reports, competes_with

Valid output example:
{
  "results": [
    {
      "chunk_id": "abc",
      "entities": [
        {"name": "NVIDIA", "type": "Company", "description": "semiconductor company"},
        {"name": "data center revenue", "type": "FinancialMetric", "description": "revenue from data center products"},
        {"name": "GPU", "type": "Product", "description": "graphics processing unit"}
      ],
      "relations": [
        {"source": "NVIDIA", "target": "data center revenue", "relation": "reports", "description": "NVIDIA reports data center revenue in financial statements", "strength": 0.9},
        {"source": "NVIDIA", "target": "GPU", "relation": "owns", "description": "NVIDIA manufactures and sells GPU products", "strength": 0.95}
      ]
    }
  ]
}

Input: """


def _get_client() -> OpenAI:
    api_key = (
        (config.LLM_API_KEY if config else "")
        or os.getenv("LLM_API_KEY", "")
        or (config.FIREWORKS_API_KEY if config else "")
        or os.getenv("FIREWORKS_API_KEY", "")
        or "not-needed"
    )
    base_url = (
        (config.LLM_BASE_URL if config else "")
        or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _parse_maybe_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    raw = raw.replace("\\n", " ").replace("\\t", " ")
    raw = re.sub(r'[\x00-\x1f]', ' ', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_extraction_payload(item: dict) -> tuple[list[dict], list[dict]]:
    entities = []
    for e in item.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        etype = str(e.get("type", "Concept")).strip()
        desc = str(e.get("description", "")).strip()
        if not name:
            continue
        entities.append({"name": name, "type": etype, "description": desc})

    relations = []
    for r in item.get("relations", []):
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rel = str(r.get("relation", "related_to")).strip()
        desc = str(r.get("description", "")).strip()
        strength = r.get("strength", 0.5)
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 0.5
        strength = max(0.0, min(1.0, strength))
        if not src or not tgt:
            continue
        relations.append({
            "source": src,
            "target": tgt,
            "relation": rel,
            "description": desc,
            "strength": strength,
        })

    return entities, relations


def _has_extraction(item: dict) -> bool:
    return bool(item.get("entities") or item.get("relations"))


def _load_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = {}
    if not checkpoint_path.exists():
        return checkpoint

    with open(checkpoint_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: ignoring malformed checkpoint line {line_no}")
                continue
            if not _has_extraction(item):
                continue
            chunk_id = item.get("chunk_id")
            if chunk_id:
                checkpoint[chunk_id] = item
    return checkpoint


def _extract_batch(batch: list[dict], model: str) -> tuple[list[dict], int]:
    client = _get_client()
    payload = [
        {"chunk_id": c.get("chunk_id", ""), "text": c.get("text", "")[:800]}
        for c in batch
    ]
    prompt_filled = ENTITY_EXTRACTION_PROMPT + json.dumps(payload, ensure_ascii=True)

    batch_results = None
    last_err = None

    for attempt in range(1, 6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract financial entities and relationships from SEC filings. Always output valid JSON.",
                    },
                    {"role": "user", "content": prompt_filled},
                ],
                temperature=0.0,
                max_tokens=4096,
                timeout=120.0,
            )
            raw_text = resp.choices[0].message.content
            if not raw_text or len(raw_text.strip()) < 10:
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                if reasoning:
                    raw_text = reasoning
            data = _parse_maybe_json(raw_text)
            if data is None:
                raise ValueError("LLM output was not valid JSON")
            batch_results = data
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = (
                ("429" in msg)
                or ("503" in msg)
                or ("resource exhausted" in msg)
                or ("rate" in msg)
                or ("unavailable" in msg)
            )
            if not retryable and not isinstance(e, ValueError):
                if attempt >= 3:
                    break
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))

    if batch_results is None:
        print(f"  WARNING: extraction failed for batch; not checkpointing {len(batch)} chunks: {last_err}")
        return [], len(batch)

    raw_results = batch_results.get("results", []) if isinstance(batch_results, dict) else []
    if not raw_results and isinstance(batch_results, list):
        raw_results = batch_results
    if not raw_results:
        print(f"  WARNING: extraction returned no results; not checkpointing {len(batch)} chunks")
        return [], len(batch)

    by_chunk = {}
    for item in raw_results:
        if isinstance(item, dict):
            cid = str(item.get("chunk_id", ""))
            if cid:
                by_chunk[cid] = item

    batch_extractions = []
    for chunk in batch:
        cid = chunk.get("chunk_id", "")
        if cid not in by_chunk:
            print(f"  WARNING: extraction response missing chunk_id {cid}; not checkpointing")
            continue
        payload_item = by_chunk[cid]
        entities, relations = _normalize_extraction_payload(payload_item)
        if not entities and not relations:
            print(f"  WARNING: empty extraction for chunk_id {cid}; not checkpointing")
            continue
        batch_extractions.append({
            "chunk_id": cid,
            "entities": entities,
            "relations": relations,
        })

    return batch_extractions, len(batch)


def extract_from_chunks(
    chunks: list[dict],
    batch_size: int = 8,
    concurrency: int = 4,
    checkpoint_path: str | Path | None = None,
    model: str | None = None,
    dry_run: bool = False,
) -> list[dict]:
    if not chunks:
        return []

    graph_dir = Path(getattr(config, "INGESTION_GRAPH_DIR", "data/graphs"))
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path
        else PROJECT_ROOT / graph_dir / "graph_extractions_checkpoint.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)

    model = (
        model
        or (config.NEO4J_ENTITY_EXTRACTION_MODEL if config else None)
        or (config.LLM_MODEL if config else None)
        or os.getenv("NEO4J_ENTITY_EXTRACTION_MODEL", DEFAULT_MODEL)
    )

    results = []
    processed = 0
    skipped = 0
    checkpointed = 0

    if dry_run:
        print(
            f"  Dry run: would process {len(chunks)} chunks "
            f"in batches of {batch_size} with concurrency {concurrency}"
        )
        return []

    to_process = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid in checkpoint:
            results.append(checkpoint[cid])
            skipped += 1
        else:
            to_process.append(chunk)

    if not to_process:
        print(f"  All {len(chunks)} chunks already in checkpoint")
        return results

    print(
        f"  Processing {len(to_process)} chunks ({skipped} skipped from checkpoint) "
        f"in batches of {batch_size} with concurrency {concurrency}"
    )

    batches = [
        to_process[batch_start:batch_start + batch_size]
        for batch_start in range(0, len(to_process), batch_size)
    ]
    max_workers = max(1, min(concurrency, len(batches)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_batch, batch, model) for batch in batches]
        with open(checkpoint_path, "a") as f:
            for future in as_completed(futures):
                try:
                    batch_extractions, batch_count = future.result()
                except Exception as exc:
                    print(f"  WARNING: extraction worker failed unexpectedly: {exc}")
                    continue
                for chunk_result in batch_extractions:
                    results.append(chunk_result)
                    f.write(json.dumps(chunk_result, ensure_ascii=False) + "\n")
                f.flush()

                processed += batch_count
                checkpointed += len(batch_extractions)
                print(
                    f"  Processed {processed}/{len(to_process)} chunks "
                    f"({len(batch_extractions)} checkpointed from last batch)"
                )

    pending = len(chunks) - len(results)
    print(
        f"  Extraction pass complete: {len(results)}/{len(chunks)} chunks available "
        f"({skipped} skipped, {checkpointed} newly checkpointed, {pending} pending retry)"
    )
    return results
