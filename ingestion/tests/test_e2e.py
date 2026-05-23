""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Test: End-to-End Pipeline                                                ║
    ║  Complete demo: clean -> parse -> chunk -> JSONL output.                 ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Validates the entire ingestion pipeline:
    1. Clean raw HTML (remove XBRL, hidden elements)
    2. Parse into typed elements
    3. Chunk with parent/child hierarchy
    4. Write to JSONL with full metadata

Run:
    python -m ingestion.tests.test_e2e [--file path/to/file.htm] [--out path/output.jsonl]

Expected Output:
    - Cleaned file path
    - Total chunk count
    - Chunk type distribution (NarrativeText, Table)
    - Output JSONL file with serialized chunks
"""

import argparse
from pathlib import Path
from dataclasses import asdict
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.core.cleaner import clean_edgar_html
from ingestion.core.chunker import chunk_filing


def main():
    """Run complete pipeline: clean -> parse -> chunk -> JSONL."""
    parser = argparse.ArgumentParser(description="End-to-end EDGAR pipeline demo")
    parser.add_argument("--file", type=str, help="Path to raw .htm filing")
    parser.add_argument("--out", type=str, default="data/processed/sample_chunks.jsonl")
    args = parser.parse_args()

    # Find first .htm file if not specified
    project_root = Path(__file__).resolve().parents[1]
    raw_path = Path(args.file) if args.file else next((project_root / "data/raw").glob("*.htm"), None)

    if not raw_path or not raw_path.exists():
        print("No raw filing found. Provide --file.")
        return

    # Step 1: Clean HTML
    cleaned_path = clean_edgar_html(raw_path)
    print(f"1. Cleaned: {cleaned_path}")

    # Step 2: Parse and chunk
    chunks = chunk_filing(cleaned_path, raw_path)
    print(f"2. Total chunks: {len(chunks)}")

    # Step 3: Count chunk types
    type_counts = {}
    for c in chunks:
        type_counts[c.element_type] = type_counts.get(c.element_type, 0) + 1
    print(f"3. Chunk types: {type_counts}")

    # Step 4: Write to JSONL
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"4. Written {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
