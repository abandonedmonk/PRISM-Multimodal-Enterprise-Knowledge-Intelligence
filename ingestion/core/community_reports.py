import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None

from ingestion.core.community_detector import (
    get_community_entities,
    get_community_relations,
    detect_communities,
)

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

COMMUNITY_REPORT_PROMPT = """You are a financial analyst synthesizing information about a semantic community extracted from SEC filings.
Return ONLY valid JSON matching without any escape characters or markdown formatting, and nothing else. The JSON should follow this schema:

{{
  "title": str,
  "summary": str,
  "key_points": [str],
  "risks": [str]
}}

Rules:
- title: 3-6 word descriptive title for this community
- summary: 3-4 sentence paragraph summarizing what this community represents
- key_points: 3-6 bullet points of the most important facts
- risks: 0-3 risk factors mentioned (or empty array if none)
- No markdown, no extra text, output starts with OBJECT and ends with OBJECT

Valid output example:

{{
  "title": "NVIDIA Data Center Operations",
  "summary": "This community centers on NVIDIA's data center business segment, which has experienced significant growth driven by AI and cloud computing demand. The company reports substantial revenue increases in this segment across fiscal years. Key products include A100 and H100 GPU architectures deployed in hyperscale data centers.",
  "key_points": [
    "Data center revenue grew 409% year-over-year in FY2024",
    "AI training and inference workloads are primary demand drivers",
    "H100 GPU architecture represents the latest product cycle"
  ],
  "risks": [
    "Potential slowdown in AI infrastructure spending",
    "Export restrictions on high-performance GPUs to China"
  ]
}}

Community id: {community_id}

Top entities in this community:
{entities_text}

Top relationships:
{relations_text}"""


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


def generate_community_report(
    community_id: str,
    client: OpenAI,
    model: str,
    checkpoint_data: dict | None = None,
) -> dict:
    if checkpoint_data and checkpoint_data.get("status") == "complete":
        return checkpoint_data

    entities = get_community_entities(community_id, top_n=20)
    relations = get_community_relations(community_id, top_n=30)

    if not entities:
        return {"community_id": community_id, "status": "skipped", "title": "", "summary": "", "key_points": [], "risks": []}

    entities_text = "\n".join(
        f"- {e['name']} ({e['type']}): {e.get('summary', '')}"
        for e in entities
    )
    relations_text = "\n".join(
        f"- {r['source']} --[{r['relation_type']}]--> {r['target']}: {r.get('description', '')}"
        for r in relations[:30]
    )

    prompt_filled = COMMUNITY_REPORT_PROMPT.format(
        community_id=community_id,
        entities_text=entities_text,
        relations_text=relations_text,
    )

    for attempt in range(1, 6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You synthesize financial community data into structured reports. Always output valid JSON."},
                    {"role": "user", "content": prompt_filled},
                ],
                temperature=0.0,
                max_tokens=4096,
                timeout=120.0,
            )
            raw = resp.choices[0].message.content.strip()
            data = _parse_maybe_json(raw)
            if data is None:
                raise ValueError("LLM output was not valid JSON")
            return {
                "community_id": community_id,
                "status": "complete",
                "title": data.get("title", ""),
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []),
                "risks": data.get("risks", []),
            }
        except Exception as e:
            msg = str(e).lower()
            retryable = (
                ("429" in msg)
                or ("503" in msg)
                or ("rate" in msg)
                or ("unavailable" in msg)
                or ("resource exhausted" in msg)
            )
            if not retryable and not isinstance(e, ValueError):
                if attempt >= 3:
                    break
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))

    return {
        "community_id": community_id,
        "status": "error",
        "title": "",
        "summary": "Failed after retries",
        "key_points": [],
        "risks": [],
    }


def generate_all_reports(
    checkpoint_path: str | Path | None = None,
    model: str | None = None,
) -> list[dict]:
    graph_dir = Path(getattr(config, "INGESTION_GRAPH_DIR", "data/graphs"))
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path
        else PROJECT_ROOT / graph_dir / "community_reports_checkpoint.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        checkpoint[item["community_id"]] = item
        except Exception:
            checkpoint = {}

    communities = detect_communities()
    if not communities:
        print("  No communities found. Run detect_communities() first.")
        return []

    client = _get_client()
    model = (
        model
        or (config.NEO4J_COMMUNITY_REPORT_MODEL if config else None)
        or (config.LLM_MODEL if config else None)
        or os.getenv("NEO4J_COMMUNITY_REPORT_MODEL", DEFAULT_MODEL)
    )

    reports = []
    for i, comm in enumerate(communities):
        cid = comm["community_id"]
        existing = checkpoint.get(cid)

        if existing:
            report = existing
            print(f"  [{i+1}/{len(communities)}] {cid} (skipped - checkpoint)")
        else:
            print(f"  [{i+1}/{len(communities)}] Generating report for {cid} ({comm['size']} nodes)...")
            report = generate_community_report(cid, client, model, existing)

            with open(checkpoint_path, "a") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")

        if report["status"] == "complete":
            _persist_report(report)

        reports.append(report)

    print(f"\n  Generated {len([r for r in reports if r['status'] == 'complete'])}/{len(reports)} community reports")
    return reports


def _persist_report(report: dict):
    try:
        from ingestion.core.graph_builder import _get_driver
    except ImportError:
        return

    try:
        driver = _get_driver()
        with driver.session() as session:
            session.run("""
                MATCH (c:Community {community_id: $cid})
                SET c.title = $title,
                    c.summary = $summary,
                    c.key_points = $key_points,
                    c.risks = $risks
            """,
                cid=report["community_id"],
                title=report.get("title", ""),
                summary=report.get("summary", ""),
                key_points=report.get("key_points", []),
                risks=report.get("risks", []),
            )
    except Exception as e:
        print(f"  WARNING: Failed to persist report to Neo4j: {e}")
