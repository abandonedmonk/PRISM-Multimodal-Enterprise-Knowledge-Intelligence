""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  EDGAR HTML Cleaner                                                        ║
    ║  Strips XBRL, hidden content, and boilerplate from SEC filings.           ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Clean raw SEC EDGAR HTML by removing interactive data (ix: tags), hidden
    elements, and boilerplate text to prepare for parsing and chunking.

Key Functions:
    clean_edgar_html(filepath)    - Remove XBRL markup & hidden elements
    build_toc_map(soup)           - Extract table of contents with section mappings
    extract_sections(soup)        - Identify section boundaries in cleaned HTML
    should_skip(text)             - Filter boilerplate and low-value text

Usage:
    cleaned_path = clean_edgar_html("NVDA_2024_10K.htm")
    # Outputs: NVDA_2024_10K_clean.htm
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ══════════════════════════════════════════════════════════════════════════════
# Section Pattern Definitions (10-K/10-Q Items)
# ══════════════════════════════════════════════════════════════════════════════

SECTION_PATTERNS = [
    (re.compile(r"^item\s*1\b", re.I), "item1", "Business Overview"),
    (re.compile(r"^item\s*1a\b", re.I), "item1a", "Risk Factors"),
    (re.compile(r"^item\s*1b\b", re.I), "item1b", "Unresolved Staff Comments"),
    (re.compile(r"^item\s*1c\b", re.I), "item1c", "Cybersecurity"),
    (re.compile(r"^item\s*2\b", re.I), "item2", "Properties"),
    (re.compile(r"^item\s*3\b", re.I), "item3", "Legal Proceedings"),
    (re.compile(r"^item\s*4\b", re.I), "item4", "Mine Safety"),
    (re.compile(r"^item\s*5\b", re.I), "item5", "Market for Common Equity"),
    (re.compile(r"^item\s*6\b", re.I), "item6", "Reserved"),
    (re.compile(r"^item\s*7\b", re.I), "item7", "MD&A"),
    (re.compile(r"^item\s*7a\b", re.I), "item7a", "Market Risk"),
    (re.compile(r"^item\s*8\b", re.I), "item8", "Financial Statements"),
    (re.compile(r"^item\s*9\b", re.I), "item9", "Accountant Disagreements"),
    (re.compile(r"^item\s*9a\b", re.I), "item9a", "Controls and Procedures"),
    (re.compile(r"^item\s*9b\b", re.I), "item9b", "Other Information"),
    (re.compile(r"^item\s*9c\b", re.I), "item9c", "Foreign Jurisdiction Disclosure"),
    (re.compile(r"^item\s*10\b", re.I), "item10", "Directors and Officers"),
    (re.compile(r"^item\s*11\b", re.I), "item11", "Executive Compensation"),
    (re.compile(r"^item\s*12\b", re.I), "item12", "Security Ownership"),
    (re.compile(r"^item\s*13\b", re.I), "item13", "Related Transactions"),
    (re.compile(r"^item\s*14\b", re.I), "item14", "Accountant Fees"),
    (re.compile(r"^item\s*15\b", re.I), "item15", "Exhibits and Schedules"),
    (re.compile(r"^item\s*16\b", re.I), "item16", "10-K Summary"),
]

# Common boilerplate phrases and legal disclaimers to filter during parsing
BOILERPLATE_PATTERNS = [
    re.compile(r"forward-looking statements", re.I),
    re.compile(r"safe harbor", re.I),
    re.compile(r"© (19|20)\d{2}", re.I),
    re.compile(r"copyright", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"this report contains", re.I),
    re.compile(r"sec\.gov", re.I),
    re.compile(r"form 10-k|form 10-q", re.I),
    re.compile(r"document and entity information", re.I),
    re.compile(r"cover page", re.I),
]

# ══════════════════════════════════════════════════════════════════════════════
# Main Cleaning Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def clean_edgar_html(filepath: str | Path) -> str:
    """Remove XBRL markup, hidden elements, and headers from SEC filing.
    
    Transformations:
      - Unwraps all ix: (iXBRL) tags
      - Removes elements with display:none CSS
      - Removes ix:header markup if present
    
    Args:
        filepath: Path to raw .htm filing file
        
    Returns:
        Path to cleaned file (same stem + _clean.htm suffix)
        
    Example:
        cleaned = clean_edgar_html("NVDA_2024_10K.htm")
        # Returns: "NVDA_2024_10K_clean.htm"
    """
    filepath = Path(filepath)
    # Parse HTML with lxml for robustness
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # ════════════════════════════════════════════════════════════════════
    # Step 1: Remove Interactive Data Elements (iXBRL)
    # ════════════════════════════════════════════════════════════════════
    # iXBRL (ix:) tags contain XBRL instance data for regulatory processing
    # They're redundant with human-readable content; unwrap to keep content
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()  # Remove tag but keep children

    # ════════════════════════════════════════════════════════════════════
    # Step 2: Remove Hidden CSS Elements
    # ════════════════════════════════════════════════════════════════════
    # SEC filings often have hidden divs for footnotes/references
    # Remove anything with display:none in style attribute
    for tag in soup.find_all(True, style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()  # Remove tag and all children

    # ════════════════════════════════════════════════════════════════════
    # Step 3: Remove iXBRL Header Block
    # ════════════════════════════════════════════════════════════════════
    # ix:header contains document metadata and is not human-readable
    ix_header = soup.find("ix:header")
    if ix_header:
        ix_header.decompose()

    # ════════════════════════════════════════════════════════════════════
    # Write cleaned HTML to output file
    # ════════════════════════════════════════════════════════════════════
    cleaned_path = str(filepath).replace(".htm", "_clean.htm")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return cleaned_path


def build_toc_map(soup: BeautifulSoup) -> dict[str, str]:
    """Extract table of contents and map anchor IDs to section metadata.
    
    Finds all links to 10-K/10-Q items and creates a mapping from HTML element
    IDs to section information. Used for section boundary detection.
    
    Args:
        soup: Parsed BeautifulSoup object of cleaned filing HTML
        
    Returns:
        Dict mapping {element_id: {anchor_id, section, toc_text}}
        
    Example:
        toc = build_toc_map(soup)
        # toc["div_42"] = {"anchor_id": "item7", "section": "MD&A", ...}
    """
    toc = {}
    # Find all table of contents links (Item 1, 1A, 7, etc.)
    toc_links = soup.find_all("a", string=re.compile(r"item\s*\d+[a-z]?", re.I))
    
    # For each TOC link, map the target div ID to section metadata
    for a_tag in toc_links:
        # Get the href (e.g., "#item7_div" or "#div_123")
        href = a_tag.get("href", "")
        # Get the link text (e.g., "Item 7 - MD&A")
        text = a_tag.get_text(strip=True)
        
        # Only process links that point to internal anchors
        if href.startswith("#"):
            # Extract the target element ID (remove # prefix)
            target_id = href[1:]
            # Match link text against known section patterns
            for pattern, anchor_id, section_label in SECTION_PATTERNS:
                if pattern.search(text):
                    # Map target div ID to section metadata
                    toc[target_id] = {
                        "anchor_id": anchor_id,
                        "section": section_label,
                        "toc_text": text,
                    }
                    # Found match, don't check remaining patterns
                    break
    return toc


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """Identify section boundaries (Item 1, 1A, 7, etc.) in filing HTML.
    
    Two-pronged approach:
      1. Use TOC links to identify explicit section div boundaries
      2. Fall back to colored span headers if TOC unavailable
    
    Args:
        soup: Parsed BeautifulSoup object of cleaned filing
        
    Returns:
        List of dicts with section metadata and DOM references
        
    Note:
        Different filings have different HTML structures. This handles both
        traditional div-based layouts and span-header layouts.
    """
    # Extract TOC mapping from document structure
    toc = build_toc_map(soup)

    # Find document body
    body = soup.find("body")
    if not body:
        return []

    # ════════════════════════════════════════════════════════════════════
    # Strategy 1: Use TOC links to identify section divs
    # ════════════════════════════════════════════════════════════════════
    # Most SEC filings have TOC links pointing to div IDs
    # Use these to identify section boundaries
    sections = []
    # Start with cover page as default
    current_section = {"anchor_id": "cover", "section": "Cover Page", "start_div": None}

    # Iterate through all divs and check if they're referenced in TOC
    for div in body.find_all("div", recursive=True):
        div_id = div.get("id", "")
        # Check if this div ID is in our TOC mapping
        if div_id in toc:
            # Save previous section if it had content
            if current_section["start_div"] is not None:
                sections.append(current_section)
            # Start new section at this div
            current_section = {
                "anchor_id": toc[div_id]["anchor_id"],
                "section": toc[div_id]["section"],
                "start_div": div,
            }

    # Don't forget the last section
    if current_section["start_div"] is not None:
        sections.append(current_section)

    # ════════════════════════════════════════════════════════════════════
    # Strategy 2: Fall back to colored span headers
    # ════════════════════════════════════════════════════════════════════
    # Some filings use colored span elements instead of TOC divs
    # Look for specific colors (#76b900 green, #0000ff blue) with Item headers
    header_spans = soup.find_all(
        "span",
        style=re.compile(r"color:#76b900|color:#0000ff", re.I),
        string=re.compile(r"^item\s*\d+[a-z]?", re.I),
    )
    span_sections = []
    for s in header_spans:
        text = s.get_text(strip=True)
        # Get parent div to use as section reference
        parent_div = s.find_parent("div")
        # Match span header text to section patterns
        for pattern, anchor_id, section_label in SECTION_PATTERNS:
            if pattern.search(text):
                span_sections.append({
                    "anchor_id": anchor_id,
                    "section": section_label,
                    "header_text": text,
                    "parent_div": parent_div,
                })
                break

    # Return TOC-based sections if found, otherwise use span-based approach
    return sections if sections else span_sections


def should_skip(text: str) -> bool:
    """Filter boilerplate, numeric-only, and trivial text elements.
    
    Avoids ingesting:
    - Very short text (<50 chars): likely headers or artifacts
    - Standalone numbers: likely formatting artifacts
    - Known boilerplate phrases: non-informative legal text
    
    Args:
        text: Text element to evaluate
        
    Returns:
        True if should be filtered out, False if should be kept
    """
    text = text.strip()
    
    # ════════════════════════════════════════════════════════════════════
    # Filter 1: Skip very short fragments
    # ════════════════════════════════════════════════════════════════════
    # Elements less than 50 chars are usually headers, footers, or artifacts
    if len(text) < 50:
        return True
    
    # ════════════════════════════════════════════════════════════════════
    # Filter 2: Skip standalone numbers
    # ════════════════════════════════════════════════════════════════════
    # Pattern matches: "100", "$1,234.56", "2.5b", "$10m"
    # These are likely formatting artifacts without semantic value
    if re.match(r"^\$?[\d,\.]+[bmk]?$", text):
        return True
    
    # ════════════════════════════════════════════════════════════════════
    # Filter 3: Skip known boilerplate phrases
    # ════════════════════════════════════════════════════════════════════
    # Check against patterns: forward-looking, safe harbor, copyright, etc.
    return any(p.search(text) for p in BOILERPLATE_PATTERNS)
