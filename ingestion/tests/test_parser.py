""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Test: EDGAR Filing Parser                                                 ║
    ║  Demo script for parsing cleaned HTML into typed elements.                ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Validates the parser module by:
    - Parsing a cleaned filing into typed elements
    - Counting and displaying element types
    - Showing section detection results
    - Displaying sample elements

Run:
    python -m ingestion.tests.test_parser [--file path/to/file_clean.htm]

Expected Output:
    - Total element count
    - Type distribution (NarrativeText, Table, ListItem, Text)
    - Elements per section
    - Sample elements with their types
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.core.cleaner import clean_edgar_html
from ingestion.core.parser import parse_filing


def main():
    """Parse a filing and display statistics on parsed elements."""
    parser = argparse.ArgumentParser(description="Test EDGAR parser")
    parser.add_argument("--file", type=str, help="Path to raw .htm filing")
    args = parser.parse_args()

    # Find first .htm file if not specified
    project_root = Path(__file__).resolve().parents[1]
    raw_path = Path(args.file) if args.file else next((project_root / "data/raw").glob("*.htm"), None)

    if not raw_path or not raw_path.exists():
        print("No raw filing found. Provide --file.")
        return

    # Step 1: Clean the filing
    cleaned_path = clean_edgar_html(raw_path)
    print(f"Cleaned: {cleaned_path}")

    # Step 2: Parse into typed elements
    parsed = parse_filing(cleaned_path)
    print(f"\nTotal parsed elements: {len(parsed)}")

    # Step 3: Count element types
    type_counts = {}
    for p in parsed:
        type_counts[p.element_type] = type_counts.get(p.element_type, 0) + 1
    print("\nElement type counts:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # Step 4: Count elements per section
    section_counts = {}
    for p in parsed:
        key = f"{p.anchor_id}: {p.section}"
        section_counts[key] = section_counts.get(key, 0) + 1
    print("\nElements per section:")
    for s, c in list(section_counts.items())[:10]:
        print(f"  {s}: {c}")

    # Step 5: Display sample elements
    print("\nFirst 10 parsed elements:")
    for i, p in enumerate(parsed[:10]):
        safe = p.text[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"  [{i}] <{p.element_type}> [{p.anchor_id}] {safe}")


if __name__ == "__main__":
    main()
