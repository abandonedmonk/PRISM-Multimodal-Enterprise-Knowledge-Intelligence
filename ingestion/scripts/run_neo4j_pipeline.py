import json
import sys
import argparse
from pathlib import Path
import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline(
    jsonl_file: str | Path | None = None,
    skip_extraction: bool = False,
    clear_graph: bool = True,
    dry_run: bool = False,
):
    processed_dir = PROJECT_ROOT / config.INGESTION_PROCESSED_DIR

    if dry_run:
        print("DRY RUN - no changes will be made\n")

    if jsonl_file:
        files = [Path(jsonl_file)]
    else:
        files = sorted(processed_dir.glob("*_chunks.jsonl"))

    if not files:
        print(f"No *_chunks.jsonl files found in {processed_dir}")
        return

    print(f"Found {len(files)} filing(s)\n")

    total_chunks = 0
    first_filing = True

    checkpoint_extractions = None
    if not dry_run and skip_extraction:
        checkpoint_path = (
            PROJECT_ROOT
            / config.INGESTION_GRAPH_DIR
            / "graph_extractions_checkpoint.json"
        )
        if checkpoint_path.exists():
            checkpoint_extractions = []
            with open(checkpoint_path) as f:
                for line_no, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"  WARNING: ignoring malformed checkpoint line {line_no}")
                            continue
                        if item.get("entities") or item.get("relations"):
                            checkpoint_extractions.append(item)
            if checkpoint_extractions:
                print(
                    f"Loaded {len(checkpoint_extractions)} checkpointed extractions "
                    f"from {checkpoint_path}"
                )
        else:
            print(f"  No checkpoint found at {checkpoint_path}")

    graph_built_from_checkpoint = False

    for jsonl_path in files:
        print(f"{'='*60}")
        print(f"Processing: {jsonl_path.name}")
        print(f"{'='*60}\n")

        chunks = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            print(f"  No chunks in {jsonl_path.name}, skipping")
            continue

        print(f"  {len(chunks)} chunks loaded")
        total_chunks += len(chunks)

        if not dry_run:
            from ingestion.core.entity_extractor import extract_from_chunks
            from ingestion.core.graph_builder import build_graph

            print(f"\n  [Step 2] Extracting entities + relations...")
            if skip_extraction:
                print("  (skipping - using existing checkpoint)")
            else:
                extractions = extract_from_chunks(
                    chunks,
                    batch_size=config.INGESTION_GRAPH_BATCH_SIZE,
                    concurrency=config.INGESTION_GRAPH_CONCURRENCY,
                )
                print(f"  Extracted from {len(extractions)} chunks")

            print(f"\n  [Step 3] Building Neo4j graph...")
            should_clear = clear_graph and first_filing
            if skip_extraction:
                if checkpoint_extractions and not graph_built_from_checkpoint:
                    build_graph(checkpoint_extractions, clear=should_clear)
                    graph_built_from_checkpoint = True
                elif graph_built_from_checkpoint:
                    print("  (skipping - graph already built once from checkpoint)")
                else:
                    print("  No checkpoint found, skipping extraction")
            else:
                build_graph(extractions, clear=should_clear)

            first_filing = False

            print()

    if not dry_run and total_chunks > 0:
        print(f"{'='*60}")
        print(f"  [Step 4] Detecting communities (Louvain)...")
        from ingestion.core.community_detector import detect_communities
        communities = detect_communities()
        print(f"  {len(communities)} communities detected")

        print(f"\n  [Step 5] Generating community reports...")
        from ingestion.core.community_reports import generate_all_reports
        reports = generate_all_reports()
        print(f"  {len([r for r in reports if r['status'] == 'complete'])} reports generated")

        print(f"\n  [Step 6] Embedding entities + reports to Neo4j vector index...")
        from ingestion.core.vector_indexer import index_entities_and_communities
        indexed = index_entities_and_communities()
        print(f"  Indexed {indexed} items")

    print(f"\n{'='*60}")
    print(f"Total chunks processed: {total_chunks}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run the Neo4j GraphRAG pipeline on filing chunks")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip entity extraction, use existing checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be processed")
    args = parser.parse_args()

    run_pipeline(
        skip_extraction=args.skip_extraction,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
