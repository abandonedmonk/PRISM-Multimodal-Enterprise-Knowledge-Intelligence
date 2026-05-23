""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Process EDGAR Filings Orchestrator                                        ║
    ║  Main pipeline: clean -> parse -> chunk -> write JSONL to data/processed.  ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Batch process all raw SEC EDGAR HTM filings through the complete pipeline:
    1. Clean HTML (remove XBRL, hidden elements)
    2. Parse into typed elements
    3. Group by section and chunk with parent/child hierarchy
    4. Write to JSONL with complete metadata

Run:
    # Process all HTM files in data/raw/
    python -m ingestion.scripts.process_all_filings
    
    # Custom paths
    python -m ingestion.scripts.process_all_filings \\
        --raw-dir data/raw \\
        --processed-dir data/processed
    
    # Reprocess files that already exist
    python -m ingestion.scripts.process_all_filings --force

Output:
    JSONL files in data/processed/ with naming: {stem}_chunks.jsonl
    Each line is a complete chunk with ~18 metadata fields

Pipeline Steps:
    1. Clean:   Remove XBRL markup, hidden elements
    2. Parse:   Partition into typed elements (NarrativeText, Table, etc.)
    3. Chunk:   Split by section, create parent/child hierarchy
    4. Write:   Serialize to JSONL
"""

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

# ══════════════════════════════════════════════════════════════════════════════
# Single Filing Processing
# ══════════════════════════════════════════════════════════════════════════════

def process_filing(raw_path: Path, processed_dir: Path, force: bool = False) -> int | None:
    """Process a single filing through the complete pipeline.
    
    Pipeline:
      1. Check if output already exists (skip unless --force)
      2. Clean HTML (remove XBRL, hidden elements)
      3. Parse into typed elements
      4. Chunk with parent/child hierarchy
      5. Write chunks to JSONL
    
    Args:
        raw_path: Path to raw .htm filing
        processed_dir: Output directory for JSONL
        force: If True, reprocess even if output exists
        
    Returns:
        Number of chunks produced, None if skipped, 0 if failed
    """
    stem = raw_path.stem
    out_file = processed_dir / f"{stem}_chunks.jsonl"

    # Skip if already processed (unless --force)
    if out_file.exists() and out_file.stat().st_size > 100 and not force:
        print(f"  SKIP: {out_file.name} already exists")
        return None

    # Step 1: Clean HTML
    print(f"  Cleaning {raw_path.name}...")
    cleaned_path = clean_edgar_html(raw_path)

    # Step 2-3: Parse and chunk
    print("  Chunking...")
    chunks = chunk_filing(cleaned_path, raw_path)

    # Check for errors
    if not chunks:
        print(f"  WARNING: 0 chunks produced from {raw_path.name}")
        return 0

    # Step 4: Write to JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"  -> {len(chunks)} chunks -> {out_file.name}")
    return len(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Batch process all filings in raw directory."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch process raw SEC filings into chunk JSONL")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw HTM filings")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory for JSONL")
    parser.add_argument("--pattern", default="*.htm", help="Glob pattern for raw files")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    # Setup directories
    raw_dir = PROJECT_ROOT / args.raw_dir
    processed_dir = PROJECT_ROOT / args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    htm_files = sorted(raw_dir.glob(args.pattern))
    if not htm_files:
        print(f"No files matching '{args.pattern}' in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(htm_files)} filing(s) in {raw_dir}\n")

    # Process each filing
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

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Processed: {processed} filings")
    print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Total chunks: {total_chunks}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
