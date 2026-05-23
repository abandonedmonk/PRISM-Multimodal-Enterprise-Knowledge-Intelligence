""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Filing Metadata Parser                                                     ║
    ║  Extracts ticker, year, quarter, CIK from filenames. Builds source URLs.   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Extract filing metadata (ticker, year, quarter, filing type) from standardized
    filenames. Maps tickers to SEC CIK numbers and builds URLs for SEC EDGAR.

Key Functions:
    parse_filename(path)         - Extract ticker, year, quarter, filing type
    build_source_url(ticker)     - Construct SEC EDGAR archive URL

Data:
    CIK_BY_TICKER               - Mapping of 10 major company tickers to CIK IDs

Usage:
    meta = parse_filename("NVDA_2024_10K.htm")
    # Returns: {ticker: "NVDA", year: 2024, filing_type: "10-K", cik: "0001045810", ...}
"""

from __future__ import annotations

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CIK Mapping for Major Tech & Finance Companies
# ══════════════════════════════════════════════════════════════════════════════

CIK_BY_TICKER = {
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

# ══════════════════════════════════════════════════════════════════════════════
# Metadata Extraction Functions
# ══════════════════════════════════════════════════════════════════════════════

def build_source_url(ticker: str, cik: str | None = None) -> str:
    """Build SEC EDGAR archive base URL from ticker or CIK.
    
    Args:
        ticker: Stock ticker symbol
        cik: SEC CIK number (optional, will look up from ticker)
        
    Returns:
        Base URL path for SEC EDGAR archives
        
    Example:
        build_source_url("NVDA")
        # Returns: "https://www.sec.gov/Archives/edgar/data/0001045810/"
    """
    cik = cik or CIK_BY_TICKER.get(ticker, "")
    if not cik:
        return ""
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/"


def parse_filename(path: str | Path) -> dict | None:
    """Extract filing metadata from standardized filename.
    
    Expected filename formats:
      - Annual:  TICKER_YEAR_FORM.htm     (e.g., NVDA_2024_10K.htm)
      - Quarterly: TICKER_QQUARTER_YEAR_FORM.htm  (e.g., AMD_Q3_2024_10Q.htm)
    
    Handles _clean suffix (from cleaning step) automatically.
    
    Args:
        path: Path to .htm filing file
        
    Returns:
        Dict with keys: {ticker, cik, year, quarter, filing_type, htm_filename, source_url}
        Returns None if filename doesn't match expected pattern
        
    Example:
        parse_filename("NVDA_2024_10K.htm")
        # Returns: {"ticker": "NVDA", "year": 2024, "quarter": None, "filing_type": "10-K", ...}
        
        parse_filename("AMD_Q3_2024_10Q.htm")
        # Returns: {"ticker": "AMD", "year": 2024, "quarter": 3, "filing_type": "10-Q", ...}
    """
    path = Path(path)
    stem = path.stem
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]

    parts = stem.split("_")
    if len(parts) < 3:
        return None

    ticker = parts[0].upper()
    quarter = None
    year = None
    filing = None

    if parts[1].startswith("Q") and len(parts) >= 4:
        try:
            quarter = int(parts[1][1:])
            year = int(parts[2])
            filing = parts[3]
        except ValueError:
            return None
    else:
        try:
            year = int(parts[1])
            filing = parts[2]
        except ValueError:
            return None

    filing_type = filing.replace("-", "")
    if filing_type == "10K":
        filing_type = "10-K"
    elif filing_type == "10Q":
        filing_type = "10-Q"

    cik = CIK_BY_TICKER.get(ticker, "")

    return {
        "ticker": ticker,
        "cik": cik,
        "year": year or 0,
        "quarter": quarter,
        "filing_type": filing_type,
        "accession_number": "",
        "htm_filename": path.name,
        "source_url": build_source_url(ticker, cik),
    }
