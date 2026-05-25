import json
import sys
import argparse
from pathlib import Path
from dataclasses import asdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.core.cleaner import clean_edgar_html
from ingestion.core.chunker import chunk_filing
import config


def process_filing(raw_path: Path, processed_dir: Path, force: bool = False) -> int | None:
    stem = raw_path.stem
    out_file = processed_dir / f"{stem}_chunks.jsonl"

    if out_file.exists() and out_file.stat().st_size > 100 and not force:
        print(f"  SKIP: {out_file.name} already exists")
        return None

    print(f"  Cleaning {raw_path.name}...")
    cleaned_path = clean_edgar_html(raw_path)

    print("  Chunking...")
    chunks = chunk_filing(cleaned_path, raw_path)

    if not chunks:
        print(f"  WARNING: 0 chunks produced from {raw_path.name}")
        return 0

    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"  -> {len(chunks)} chunks -> {out_file.name}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Batch process raw SEC filings into chunk JSONL")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / config.INGESTION_RAW_DIR
    processed_dir = PROJECT_ROOT / config.INGESTION_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    htm_files = sorted(raw_dir.glob(config.INGESTION_FILINGS_GLOB))
    if not htm_files:
        print(f"No files matching '{config.INGESTION_FILINGS_GLOB}' in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(htm_files)} filing(s) in {raw_dir}\n")

    total_chunks = 0
    processed = 0
    skipped = 0

    for i, htm_file in enumerate(htm_files, 1):
        print(f"[{i}/{len(htm_files)}] {htm_file.name}")
        result = process_filing(htm_file, processed_dir, force=args.force)
        if result is None:
            skipped += 1
        else:
            total_chunks += result
            processed += 1

    print(f"\n{'='*50}")
    print(f"  Processed: {processed} filings")
    print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Total chunks: {total_chunks}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
