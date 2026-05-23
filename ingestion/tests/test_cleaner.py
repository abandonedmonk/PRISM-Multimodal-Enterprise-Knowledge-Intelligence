""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Test: EDGAR Cleaner                                                       ║
    ║  Demo/test script for HTML cleaning and TOC extraction.                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Validates the cleaner module by:
    - Cleaning a sample filing (removes XBRL, hidden elements)
    - Extracting and displaying table of contents
    - Extracting and displaying section boundaries

Run:
    python -m ingestion.tests.test_cleaner [--file path/to/file.htm]

Expected Output:
    - Cleaned file size before/after
    - TOC entries (Item 1, 1A, 7, 8, etc.)
    - Section boundaries with anchor IDs
"""

import argparse
from pathlib import Path
import sys
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# ══════════════════════════════════════════════════════════════════════════════
# Setup: Path Resolution and Imports
# ══════════════════════════════════════════════════════════════════════════════
# Ensure PROJECT_ROOT is in path for imports to work correctly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import cleaner functions from ingestion.core
from ingestion.core.cleaner import clean_edgar_html, build_toc_map, extract_sections

# Suppress BeautifulSoup XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def main():
    """Run cleaner test: clean filing, extract TOC and sections.
    
    Shows:
    - File size reduction after cleaning
    - Table of contents entries
    - Section extraction results
    """
    # ════════════════════════════════════════════════════════════════════
    # Parse command-line arguments
    # ════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(
        description="Test EDGAR cleaner: validate cleaning, TOC extraction"
    )
    parser.add_argument("--file", type=str, help="Path to raw .htm filing")
    args = parser.parse_args()

    # ════════════════════════════════════════════════════════════════════
    # Find input file
    # ════════════════════════════════════════════════════════════════════
    # Use --file if provided, otherwise find first .htm in data/raw
    project_root = Path(__file__).resolve().parents[2]
    raw_path = Path(args.file) if args.file else next(
        (project_root / "data/raw").glob("*.htm"), None
    )

    # Validate file exists
    if not raw_path or not raw_path.exists():
        print("No raw filing found. Provide --file.")
        return

    # ════════════════════════════════════════════════════════════════════
    # Step 1: Clean the filing
    # ════════════════════════════════════════════════════════════════════
    print(f"Processing: {raw_path.name}")
    cleaned_path = clean_edgar_html(raw_path)
    print(f"Cleaned: {cleaned_path}")

    # ════════════════════════════════════════════════════════════════════
    # Step 2: Show file size reduction
    # ════════════════════════════════════════════════════════════════════
    # Demonstrates effectiveness of XBRL/hidden element removal
    raw_size = raw_path.stat().st_size
    clean_size = Path(cleaned_path).stat().st_size
    pct = clean_size / raw_size if raw_size else 0
    print(f"Size: {raw_size:,} -> {clean_size:,} ({pct:.1%} of original)")

    # ════════════════════════════════════════════════════════════════════
    # Step 3: Parse cleaned HTML and extract TOC
    # ════════════════════════════════════════════════════════════════════
    with open(cleaned_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Build table of contents mapping from TOC links
    toc = build_toc_map(soup)
    print(f"\nTOC MAP ({len(toc)} entries):")
    # Display first 10 TOC entries (truncate long IDs for readability)
    for div_id, info in list(toc.items())[:10]:
        short_id = div_id[:25]  # Truncate long element IDs
        aid = info["anchor_id"]
        sec = info["section"]
        print(f"  div#{short_id}... -> {aid}: {sec}")

    # ════════════════════════════════════════════════════════════════════
    # Step 4: Extract section boundaries
    # ════════════════════════════════════════════════════════════════════
    # Uses TOC-based or span-header-based detection
    sections = extract_sections(soup)
    print(f"\nSECTIONS ({len(sections)} found):")
    # Display first 10 sections
    for s in sections[:10]:
        aid = s["anchor_id"]
        sec = s["section"]
        print(f"  {aid}: {sec}")

if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════════════
    # Entry point: Run cleaner test
    # ════════════════════════════════════════════════════════════════════
    # Can be run as: python -m ingestion.tests.test_cleaner
    # Or with custom file: python -m ingestion.tests.test_cleaner --file path/to/file.htm
    main()
