""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Table Summarizer (LLM)                                                    ║
    ║  Summarizes financial table markdown using an LLM API (optional step).     ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Optional enhancement: Use LLM to summarize financial tables in chunks for
    better semantic understanding during embedding and retrieval.

Run:
    export FIREWORKS_API_KEY="your-api-key"
    python -m ingestion.scripts.summarize_tables --processed-dir data/processed
    
    # Dry run to preview summaries
    python -m ingestion.scripts.summarize_tables --dry-run

Configuration:
    API Key: FIREWORKS_API_KEY env var (or specify --api-key-env)
    API Endpoint: https://api.fireworks.ai/inference/v1 (default)
    Model: accounts/fireworks/models/gpt-4o-mini (default)

Updates:
    Modifies chunks' "text" field with LLM summary if table_markdown exists
"""

import json
import argparse
import time
from pathlib import Path
import sys
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# LLM Summarization
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = (
    "Summarize this financial table in 2-3 sentences. "
    "Preserve all exact numbers, percentages, and dollar amounts. "
    "Do not add any information not in the table.\n\n"
    "Table:\n{markdown}"
)


def summarize_table(client: OpenAI, model: str, markdown: str, max_retries: int = 3) -> str:
    """Summarize a table using LLM with retry logic.
    
    Args:
        client: OpenAI-compatible API client
        model: Model identifier (e.g., gpt-4o-mini)
        markdown: Markdown table to summarize
        max_retries: Number of retry attempts on failure
        
    Returns:
        Summarized text (2-3 sentences) or empty string if failed after retries
    """
    prompt = SUMMARY_PROMPT.format(markdown=markdown)
    # Retry on API failures (rate limiting, transient errors)
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
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"    FAILED after {max_retries} retries: {e}")
                return ""


def process_jsonl(
    jsonl_path: Path,
    client: OpenAI,
    model: str,
    dry_run: bool = False,
) -> int:
    """Summarize all table chunks in a JSONL file.
    
    Args:
        jsonl_path: Path to chunks JSONL file
        client: OpenAI-compatible API client
        model: Model to use for summarization
        dry_run: If True, show previews without actually summarizing
        
    Returns:
        Count of tables actually summarized (0 in dry-run)
    """
    # Load all chunks from JSONL
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # Filter to table chunks with markdown
    table_chunks = [c for c in chunks if c.get("element_type") == "Table" and c.get("table_markdown")]
    if not table_chunks:
        print(f"  No table chunks with markdown in {jsonl_path.name}")
        return 0

    print(f"  {len(table_chunks)} table(s) to summarize in {jsonl_path.name}")

    # Dry-run: show previews without processing
    if dry_run:
        for tc in table_chunks[:3]:
            print(f"    Preview: {tc['table_markdown'][:100]}...")
        return 0

    # Summarize each table
    summarized = 0
    for i, tc in enumerate(table_chunks):
        md = tc["table_markdown"]
        summary = summarize_table(client, model, md)
        if summary:
            tc["text"] = summary  # Update chunk text with summary
            summarized += 1
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(table_chunks)} done")

    # Write updated chunks back to JSONL
    out_path = jsonl_path
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"  Summarized {summarized}/{len(table_chunks)} tables -> {out_path.name}")
    return summarized


def main():
    """Summarize all table chunks using LLM."""
    parser = argparse.ArgumentParser(description="Summarize table chunks using LLM for better embeddings")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with JSONL files")
    parser.add_argument("--api-key-env", default="FIREWORKS_API_KEY", help="Env var name for API key")
    parser.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    parser.add_argument("--model", default="accounts/fireworks/models/gpt-4o-mini")
    parser.add_argument("--file", type=str, help="Process single JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be summarized")
    args = parser.parse_args()

    import os
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        print(f"ERROR: Set {args.api_key_env} environment variable")
        return

    client = OpenAI(api_key=api_key or "dummy", base_url=args.base_url)
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / args.processed_dir

    if args.file:
        files = [Path(args.file)]
    else:
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
