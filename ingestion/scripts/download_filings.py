""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Download SEC EDGAR Filings                                                ║
    ║  Fetches 10-K and 10-Q HTM files from SEC EDGAR API into data/raw.        ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Download recent 10-K and 10-Q filings for major public companies from
    SEC EDGAR API. Rate-limited to respect SEC servers (0.15s between requests).

Run:
    # Download default companies (NVDA, AMD, INTC, AAPL, MSFT, GOOGL, META, AMZN, TSLA, JPM)
    python -m ingestion.scripts.download_filings --output-dir data/raw
    
    # Download specific companies
    python -m ingestion.scripts.download_filings --tickers NVDA AMD --output-dir data/raw
    
    # Dry run (show what would be downloaded)
    python -m ingestion.scripts.download_filings --dry-run

Output:
    HTM files in data/raw/ with naming convention: TICKER_YEAR_FORM.htm
    (e.g., NVDA_2024_10K.htm, AMD_Q3_2024_10Q.htm)

Rate Limiting:
    All requests include 0.15s delay to comply with SEC guidelines
"""

import requests
import json
import time
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# SEC API Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_HEADERS = {
    "User-Agent": "PRISM/1.0 research@prism.ai",
    "Accept-Encoding": "gzip, deflate",
}

DOWNLOAD_HEADERS = {
    "User-Agent": "PRISM/1.0 research@prism.ai",
    "Accept-Encoding": "gzip, deflate",
}

# Mapping of ticker symbols to SEC CIK identifiers
COMPANIES = {
    "NVDA": "0001045810",
    "AMD": "0000002488",
    "INTC": "0000050863",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "JPM": "0000019617",
}

# Filing specifications: what to download for each company
FILING_SPECS = [
    {"form": "10-K", "label": "2024_10K", "selector": "report_year", "year": 2024},
    {"form": "10-K", "label": "2023_10K", "selector": "report_year", "year": 2023},
    {"form": "10-Q", "label": "Q4_2024_10Q", "selector": "report_quarter", "year": 2024, "quarter": 4},
    {"form": "10-Q", "label": "Q3_2024_10Q", "selector": "report_quarter", "year": 2024, "quarter": 3},
]

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def fetch_submissions(cik: str) -> dict:
    """Fetch company submission metadata from SEC API.
    
    Args:
        cik: SEC CIK identifier (with or without leading zeros)
        
    Returns:
        Parsed JSON with filing history and metadata
        
    Raises:
        requests.RequestException: If API request fails
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=API_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _date_to_quarter(date_str: str) -> tuple[int, int]:
    """Convert date string YYYY-MM-DD to (year, quarter).
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        Tuple of (year, quarter) where quarter is 1-4
    """
    year = int(date_str[:4])
    month = int(date_str[5:7])
    return year, (month - 1) // 3 + 1


def find_filing(submissions: dict, spec: dict) -> dict | None:
    """Find a specific filing in submission history matching spec.
    
    Searches for 10-K or 10-Q filings by year/quarter.
    
    Args:
        submissions: Company submission data from SEC API
        spec: Filing specification dict with form, year, quarter, selector
        
    Returns:
        Filing dict with accessionNumber, filingDate, primaryDocument
        or None if not found
    """
    recent = submissions["filings"]["recent"]
    form = spec["form"]
    selector = spec["selector"]

    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        report_date = recent["reportDate"][i]
        if not report_date:
            continue
        ry, rq = _date_to_quarter(report_date)

        if selector == "report_year":
            if ry == spec["year"]:
                return {
                    "accessionNumber": recent["accessionNumber"][i],
                    "filingDate": recent["filingDate"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                    "reportDate": report_date,
                }
        elif selector == "report_quarter":
            if ry == spec["year"] and rq == spec["quarter"]:
                return {
                    "accessionNumber": recent["accessionNumber"][i],
                    "filingDate": recent["filingDate"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                    "reportDate": report_date,
                }
    return None


def download_filing(cik: str, accession_number: str, filename: str, outpath: Path) -> bool:
    """Download HTM filing from SEC EDGAR archives.
    
    Args:
        cik: SEC CIK identifier
        accession_number: SEC accession number (with dashes)
        filename: Primary document filename (e.g. "0001045810-24-000042.htm")
        outpath: Where to save the downloaded file
        
    Returns:
        True if successful, False otherwise
    """
    acc_clean = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{filename}"
    try:
        r = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=60)
        r.raise_for_status()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_bytes(r.content)
        print(f"  Downloaded {len(r.content):,} bytes -> {outpath.name}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Download SEC filings for specified companies with rate limiting."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download SEC EDGAR filings for PRISM")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(COMPANIES.keys()),
        help="Company tickers to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for downloaded filings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    # Setup output directory
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate and normalize tickers
    tickers = [t.upper() for t in args.tickers if t.upper() in COMPANIES]
    if not tickers:
        print("No valid tickers found. Available:", list(COMPANIES.keys()))
        sys.exit(1)

    # Progress tracking
    total = len(tickers) * len(FILING_SPECS)
    downloaded = 0
    failed = 0
    skipped = 0

    # Download filings for each company
    for ticker in tickers:
        cik = COMPANIES[ticker]
        print(f"\n{'='*60}")
        print(f"  {ticker} (CIK: {cik})")
        print(f"{'='*60}")

        # Fetch submission history
        try:
            submissions = fetch_submissions(cik)
        except Exception as e:
            print(f"  FAILED to fetch submissions: {e}")
            failed += len(FILING_SPECS)
            continue

        time.sleep(0.15)  # Rate limit: 0.15s between API requests

        # Process each filing specification (10-K, 10-Q for different years/quarters)
        for spec in FILING_SPECS:
            label = spec["label"]

            # Find this specific filing in the submission history
            filing = find_filing(submissions, spec)
            if not filing:
                print(f"  {label}: NOT FOUND")
                skipped += 1
                continue

            # Output path: TICKER_LABEL.htm (e.g., NVDA_2024_10K.htm)
            outpath = output_dir / f"{ticker}_{label}.htm"

            # Skip if already downloaded
            if outpath.exists() and outpath.stat().st_size > 1000:
                print(f"  {label}: already exists ({outpath.stat().st_size:,} bytes), skipping")
                skipped += 1
                continue

            # Show what would be downloaded in dry-run mode
            if args.dry_run:
                print(
                    f"  {label}: would download {filing['primaryDocument']} "
                    f"(filed {filing['filingDate']})"
                )
                continue

            # Download the filing
            ok = download_filing(
                cik,
                filing["accessionNumber"],
                filing["primaryDocument"],
                outpath,
            )
            if ok:
                downloaded += 1
            else:
                failed += 1

            time.sleep(0.15)  # Rate limit between downloads

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {downloaded} downloaded, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
