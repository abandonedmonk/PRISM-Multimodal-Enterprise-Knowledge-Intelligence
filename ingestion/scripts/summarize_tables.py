import json
import argparse
import time
from pathlib import Path
import sys
from openai import OpenAI
import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PROMPT = (
    "Summarize this financial table in 2-3 sentences. "
    "Preserve all exact numbers, percentages, and dollar amounts. "
    "Do not add any information not in the table.\n\n"
    "Table:\n{markdown}"
)


def summarize_table(client: OpenAI, model: str, markdown: str, max_retries: int = 3) -> str:
    prompt = SUMMARY_PROMPT.format(markdown=markdown)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED after {max_retries} retries: {e}")
                return ""


def process_jsonl(
    jsonl_path: Path,
    client: OpenAI,
    model: str,
    dry_run: bool = False,
) -> int:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    table_chunks = [c for c in chunks if c.get("element_type") == "Table" and c.get("table_markdown")]
    if not table_chunks:
        print(f"  No table chunks with markdown in {jsonl_path.name}")
        return 0

    print(f"  {len(table_chunks)} table(s) to summarize in {jsonl_path.name}")

    if dry_run:
        for tc in table_chunks[:3]:
            print(f"    Preview: {tc['table_markdown'][:100]}...")
        return 0

    summarized = 0
    for i, tc in enumerate(table_chunks):
        md = tc["table_markdown"]
        summary = summarize_table(client, model, md)
        if summary:
            tc["text"] = summary
            summarized += 1
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(table_chunks)} done")

    out_path = jsonl_path
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"  Summarized {summarized}/{len(table_chunks)} tables -> {out_path.name}")
    return summarized


def main():
    parser = argparse.ArgumentParser(description="Summarize table chunks using LLM for better embeddings")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be summarized")
    args = parser.parse_args()

    import os
    api_key = (
        os.environ.get("LLM_API_KEY", "")
        or os.environ.get("FIREWORKS_API_KEY", "")
        or "not-needed"
    )
    if not api_key and not args.dry_run:
        print("ERROR: Set LLM_API_KEY or FIREWORKS_API_KEY environment variable")
        return

    client = OpenAI(api_key=api_key, base_url=config.LLM_BASE_URL)
    processed_dir = PROJECT_ROOT / config.INGESTION_PROCESSED_DIR

    files = sorted(processed_dir.glob("*_chunks.jsonl"))
    if not files:
        print("No JSONL files found")
        return

    total = 0
    for jsonl_file in files:
        count = process_jsonl(jsonl_file, client, args.model, args.dry_run)
        total += count

    print(f"\nTotal tables summarized: {total}")


if __name__ == "__main__":
    main()
