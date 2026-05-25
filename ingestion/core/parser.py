""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  EDGAR Filing Parser                                                       ║
    ║  Partitions cleaned HTML into typed elements (Text, Table, ListItem, etc). ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convert cleaned SEC EDGAR HTML into structured, typed elements using the
    unstructured library. Preserves section and element type information for
    downstream chunking and RAG.

Key Functions:
    parse_filing(cleaned_path)    - Main parser: converts HTML to typed elements
    _detect_section(text)         - Identify 10-K/10-Q section from text
    
Key Classes:
    ParsedElement                 - Data class for text elements with metadata

Usage:
    elements = parse_filing("NVDA_2024_10K_clean.htm")
    # Returns: list[ParsedElement] with element_type, section, anchor_id
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Text, NarrativeText, Table, ListItem, Image
import warnings

from ingestion.core.cleaner import SECTION_PATTERNS, should_skip, build_toc_map

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_RE = re.compile(r"^item\s*\d+[a-z]?\.", re.I)

# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedElement:
    """Text element extracted from filing with section and type information.
    
    Attributes:
        text: Raw text content of element
        element_type: Type name (NarrativeText, Table, ListItem, etc.)
        section: Human-readable section name (e.g. "MD&A")
        anchor_id: Machine-readable section ID (e.g. "item7")
        metadata: Dict of additional metadata from unstructured library
    """
    # Element content
    text: str
    # Type from unstructured library (NarrativeText, Table, ListItem, Image)
    element_type: str
    # Section display name (MD&A, Risk Factors, etc.)
    section: str = "Cover Page"
    # Section identifier for metadata (item7, item1a, etc.)
    anchor_id: str = "cover"
    # Additional metadata from unstructured extraction
    metadata: dict = field(default_factory=dict)
    # Raw HTML for Table elements (used for image rendering)
    raw_html: str = ""


def _detect_section(text: str) -> tuple[str, str] | None:
    """Identify 10-K/10-Q section name from text content.
    
    Args:
        text: Text to check against section patterns
        
    Returns:
        Tuple of (anchor_id, section_label) if matched, else None
        
    Example:
        _detect_section("Item 7. Management's Discussion and Analysis")
        # Returns: ("item7", "MD&A")
    """
    # Iterate through regex patterns imported from cleaner module
    for pattern, anchor_id, section_label in SECTION_PATTERNS:
        # Try to match section header pattern in the text
        if pattern.search(text):
            # Return section identifier and human-readable label
            return anchor_id, section_label
    # No section pattern matched
    return None


def _extract_table_html_from_soup(soup: BeautifulSoup, text: str) -> str:
    """Extract raw HTML table from BeautifulSoup for Table elements.
    
    Searches the soup for a table whose text content matches the element text.
    Returns the outerHTML of the matched table.
    
    Args:
        soup: Parsed BeautifulSoup object
        text: Text content of the table element
        
    Returns:
        Raw HTML string of the matched table, or empty string if not found
    """
    for table in soup.find_all("table"):
        if text.strip() in table.get_text() or table.get_text().strip() in text:
            return str(table)
    for table in soup.find_all("table"):
        if len(table.get_text().strip()) > 20:
            sim = len(set(table.get_text().split()) & set(text.split())) / max(len(set(text.split())), 1)
            if sim > 0.7:
                return str(table)
    return ""


def parse_filing(cleaned_path: str | Path) -> list[ParsedElement]:
    """Parse cleaned HTML into typed elements with section information.
    
    Uses unstructured library to partition HTML into typed elements, then
    annotates each element with section and anchor information by matching
    against known 10-K/10-Q section headers.
    
    Args:
        cleaned_path: Path to cleaned _clean.htm file
        
    Returns:
        List of ParsedElement objects with text, type, section, anchor_id
        
    Example:
        elements = parse_filing("NVDA_2024_10K_clean.htm")
        # Returns ~700-800 elements per filing
    """
    cleaned_path = Path(cleaned_path)
    
    elements = partition_html(
        filename=str(cleaned_path),
        skip_headers_and_footers=True,
        include_metadata=True,
    )

    with open(cleaned_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    _ = build_toc_map(soup)

    parsed: list[ParsedElement] = []
    current_anchor = "cover"
    current_section = "Cover Page"

    for el in elements:
        el_type = type(el).__name__
        text = str(el).strip()

        section_hit = _detect_section(text)
        if section_hit:
            current_anchor, current_section = section_hit
            if el_type == "Text" and len(text) < 80:
                continue

        if should_skip(text):
            continue

        if el_type == "Image":
            continue

        raw_html = ""
        if el_type == "Table":
            raw_html = _extract_table_html_from_soup(soup, text)

        parsed.append(ParsedElement(
            text=text,
            element_type=el_type,
            section=current_section,
            anchor_id=current_anchor,
            raw_html=raw_html,
        ))

    return parsed
