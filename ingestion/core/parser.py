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
    
    # ════════════════════════════════════════════════════════════════════
    # Step 1: Partition HTML into typed elements
    # ════════════════════════════════════════════════════════════════════
    # unstructured library handles detection of Text, NarrativeText, Table,
    # ListItem, Image based on HTML structure and CSS classes
    elements = partition_html(
        filename=str(cleaned_path),
        skip_headers_and_footers=True,  # Ignore page headers/footers
        include_metadata=True,
    )

    # Build TOC map for reference (may be used for anchor validation)
    with open(cleaned_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    _ = build_toc_map(soup)

    # ════════════════════════════════════════════════════════════════════
    # Step 2: Iterate through elements, annotate with section info
    # ════════════════════════════════════════════════════════════════════
    parsed: list[ParsedElement] = []
    current_anchor = "cover"      # Current section anchor ID
    current_section = "Cover Page" # Current section display name

    for el in elements:
        # Get element type (Text, NarrativeText, Table, ListItem, Image)
        el_type = type(el).__name__
        # Extract text content and strip whitespace
        text = str(el).strip()

        # ════════════════════════════════════════════════════════════════
        # Check if this element starts a new section
        # ════════════════════════════════════════════════════════════════
        section_hit = _detect_section(text)
        if section_hit:
            # Update current section anchor and label
            current_anchor, current_section = section_hit
            # Skip short section headers themselves (they're just labels)
            if el_type == "Text" and len(text) < 80:
                continue

        # ════════════════════════════════════════════════════════════════
        # Filter out boilerplate and low-value content
        # ════════════════════════════════════════════════════════════════
        # Uses should_skip() from cleaner module (checks length, pattern)
        if should_skip(text):
            continue

        # ════════════════════════════════════════════════════════════════
        # Skip image elements (no OCR/vision extraction yet)
        # ════════════════════════════════════════════════════════════════
        if el_type == "Image":
            continue

        # ════════════════════════════════════════════════════════════════
        # Add element to parsed list with current section metadata
        # ════════════════════════════════════════════════════════════════
        parsed.append(ParsedElement(
            text=text,
            element_type=el_type,
            section=current_section,
            anchor_id=current_anchor,
        ))

    return parsed
