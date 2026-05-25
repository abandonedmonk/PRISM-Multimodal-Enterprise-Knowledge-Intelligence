""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  EDGAR Filing Chunker                                                      ║
    ║  Groups parsed elements by section, chunks narrative text, extracts tables.║
    ║  Produces Chunk objects with parent/child hierarchy for RAG.              ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convert parsed elements into RAG-ready chunks with parent/child hierarchy,
    preserving section context and metadata. Handles both narrative text
    (hierarchical split) and tables (markdown extraction).

Key Functions:
    chunk_filing(cleaned_path, raw_path)  - Main orchestrator: parse -> chunk
    _group_by_section(elements)            - Group elements by filing section
    _extract_table_markdown(text)          - Convert HTML tables to markdown
    _ticker_to_company(ticker)             - Map ticker symbol to company name

Key Classes:
    Chunk                                  - RAG-ready chunk with full metadata

Configuration:
    parent_splitter: 3500 tokens, 300 overlap  (larger context)
    child_splitter:  1024 tokens, 256 overlap  (dense embeddings)

Usage:
    chunks = chunk_filing("NVDA_2024_10K_clean.htm", "NVDA_2024_10K.htm")
    # Returns: list[Chunk] with parent_text, table_markdown, company, ticker, etc.
"""

import os
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

try:
    import config as _config
    _TABLE_IMAGES_DIR = getattr(_config, "TABLE_IMAGES_DIR", "data/processed/tables")
except Exception:
    _TABLE_IMAGES_DIR = "data/processed/tables"

from ingestion.core.parser import ParsedElement, parse_filing
from ingestion.core.metadata import parse_filename, build_source_url

try:
    from ingestion.core.table_renderer import render_table_to_image
    TABLE_RENDERER_AVAILABLE = True
except ImportError:
    TABLE_RENDERER_AVAILABLE = False

try:
    from ingestion.core.vision_extractor import describe_table_image
    VISION_EXTRACTOR_AVAILABLE = True
except ImportError:
    VISION_EXTRACTOR_AVAILABLE = False

COVER_ANCHOR = "cover"

# Sections to exclude from chunking (boilerplate or repetitive)
SKIP_ANCHORS = {
    "cover",
    "item4",
    "item6",
    "item9",
    "item9b",
    "item9c",
    "item10",
    "item11",
    "item14",
    "item16",
}

# ══════════════════════════════════════════════════════════════════════════════
# Text Splitting Configuration
# ══════════════════════════════════════════════════════════════════════════════
# Parent chunks: Large context windows for retrieval (avoid over-chunking)
# Child chunks: Dense vectors for embedding and search

# Parent splitter: 3500 token chunks with 300 token overlap
# Used to create large context windows for RAG retrieval
# Larger size preserves financial statement coherence
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Child splitter: 1024 token chunks with 256 token overlap
# Used for dense vector embeddings and semantic search
# Smaller size enables precise matching and efficient embedding
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass
class Chunk:
    """RAG-ready chunk with hierarchical metadata for document retrieval.
    
    Attributes:
        text: Child chunk text (dense, 1024 tokens)
        parent_text: Parent chunk text (broader context, 3500 tokens)
        table_markdown: Markdown representation of table (if Table element)
        
        company: Company name (derived from ticker)
        ticker: Stock ticker symbol
        cik: SEC CIK identifier
        year: Fiscal year of filing
        quarter: Fiscal quarter (if 10-Q), None if 10-K
        filing_type: Form type (10-K, 10-Q)
        accession_number: SEC accession number
        
        section: Section label (e.g. "MD&A")
        anchor_id: Section ID (e.g. "item7")
        element_type: Type of element (NarrativeText, Table)
        
        chunk_index: Ordinal position in all chunks from this filing
        parent_chunk_id: UUID linking child to parent chunk
        chunk_id: Unique identifier for this chunk
        source_url: Base SEC EDGAR URL for filing
        htm_filename: Original input filename
    """
    text: str
    parent_text: str = ""
    table_markdown: str = ""
    table_image_path: str = ""
    vision_description: str = ""
    company: str = ""
    ticker: str = ""
    cik: str = ""
    year: int = 0
    quarter: int | None = None
    filing_type: str = ""
    accession_number: str = ""
    section: str = ""
    anchor_id: str = ""
    element_type: str = ""
    chunk_index: int = 0
    parent_chunk_id: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_url: str = ""
    htm_filename: str = ""


def _extract_table_markdown(text: str) -> str:
    """Convert HTML table to Markdown for better readability in embeddings.
    
    Args:
        text: HTML or raw table text
        
    Returns:
        Markdown-formatted table string, or empty string if conversion fails
    """
    try:
        # Only attempt parsing if text contains table markers
        dfs = pd.read_html(text, flavor="lxml") if "<table" in text or "|" in text else []
        # If parsing succeeded, convert first DataFrame to markdown
        if dfs:
            return dfs[0].to_markdown(index=False)
    except Exception:
        # Silently fail if table parsing doesn't work (some HTML may be malformed)
        pass
    return ""


def _group_by_section(elements: list[ParsedElement]) -> list[tuple[str, str, list[ParsedElement]]]:
    """Group parsed elements by filing section anchor.
    
    Organizes elements into sections while skipping boilerplate anchors.
    
    Args:
        elements: List of parsed elements from parse_filing()
        
    Returns:
        List of (anchor_id, section_label, elements) tuples
    """
    groups = []
    current_anchor = None
    current_section = None
    current_elements = []

    # Iterate through elements, accumulating them by anchor
    for el in elements:
        # When anchor changes, save previous group if it's not a boilerplate section
        if el.anchor_id != current_anchor:
            if current_elements and current_anchor not in SKIP_ANCHORS:
                groups.append((current_anchor, current_section, current_elements))
            # Start a new section group
            current_anchor = el.anchor_id
            current_section = el.section
            current_elements = []
        # Add element to current section
        current_elements.append(el)

    # Don't forget to append the last group
    if current_elements and current_anchor not in SKIP_ANCHORS:
        groups.append((current_anchor, current_section, current_elements))

    return groups


def chunk_filing(cleaned_path: str | Path, raw_path: str | Path | None = None) -> list[Chunk]:
    """Parse and chunk a filing into RAG-ready chunks with metadata.
    
    Pipeline:
      1. Parse cleaned HTML into typed elements
      2. Group elements by section (skip boilerplate sections)
      3. Extract file metadata (ticker, year, CIK, etc.)
      4. Split narrative text into parent/child hierarchy
      5. Extract table markdowns
      6. Return Chunk objects with full metadata
    
    Args:
        cleaned_path: Path to _clean.htm file
        raw_path: Path to original .htm file (for metadata extraction)
        
    Returns:
        List[Chunk]: ~800-1000 chunks per typical 10-K with metadata
        
    Example:
        chunks = chunk_filing(
            "NVDA_2024_10K_clean.htm",
            "NVDA_2024_10K.htm"
        )
        # chunks[0].company = "NVIDIA"
        # chunks[0].section = "MD&A"
        # chunks[0].parent_text has ~3500 tokens
    """
    cleaned_path = Path(cleaned_path)
    raw_path = Path(raw_path) if raw_path else cleaned_path

    # Extract metadata from filename (ticker, year, quarter, filing type, CIK, etc.)
    file_meta = parse_filename(raw_path)
    if not file_meta:
        file_meta = parse_filename(cleaned_path)
    # If parsing still failed, use defaults
    if not file_meta:
        file_meta = {
            "ticker": "",
            "cik": "",
            "year": 0,
            "quarter": None,
            "filing_type": "",
            "htm_filename": cleaned_path.name,
            "source_url": "",
        }

    # Step 1: Parse HTML into typed elements (NarrativeText, Table, ListItem, etc.)
    elements = parse_filing(cleaned_path)
    # Step 2: Group elements by their anchor ID (section), filtering boilerplate
    groups = _group_by_section(elements)

    chunks = []
    chunk_idx = 0

    # Process each section separately
    for anchor_id, section, section_elements in groups:
        # Separate elements by type for different handling
        narrative_texts = []
        table_texts = []
        list_items = []

        for el in section_elements:
            if el.element_type == "Table":
                table_texts.append(el)
            elif el.element_type == "ListItem":
                list_items.append(el)
            elif el.element_type in ("NarrativeText", "Text"):
                narrative_texts.append(el)

        # ════════════════════════════════════════════════════════════════════
        # Process Narrative Text with Parent/Child Hierarchy
        # ════════════════════════════════════════════════════════════════════
        # Combine all narrative text and list items, preserving section flow
        combined_text = "\n\n".join(el.text for el in narrative_texts + list_items)
        if combined_text.strip():
            # First split into parent chunks (3500 tokens, broader context)
            parent_chunks = parent_splitter.split_text(combined_text)
            for parent_text in parent_chunks:
                # Then split each parent into child chunks (1024 tokens, dense embeddings)
                child_chunks = child_splitter.split_text(parent_text)
                # Use same parent_id to link all children to this parent
                parent_id = str(uuid.uuid4())

                for child_text in child_chunks:
                    # Create chunk with all filing metadata
                    chunks.append(Chunk(
                        text=child_text,
                        parent_text=parent_text,
                        company=_ticker_to_company(file_meta.get("ticker", "")),
                        ticker=file_meta.get("ticker", ""),
                        cik=file_meta.get("cik", ""),
                        year=file_meta.get("year", 0),
                        quarter=file_meta.get("quarter"),
                        filing_type=file_meta.get("filing_type", ""),
                        accession_number=file_meta.get("accession_number", ""),
                        section=section,
                        anchor_id=anchor_id,
                        element_type="NarrativeText",
                        chunk_index=chunk_idx,
                        parent_chunk_id=parent_id,
                        source_url=file_meta.get("source_url") or build_source_url(
                            file_meta.get("ticker", ""),
                            file_meta.get("cik", ""),
                        ),
                        htm_filename=file_meta.get("htm_filename", ""),
                    ))
                    chunk_idx += 1

        # ════════════════════════════════════════════════════════════════════
        # Process Tables Separately
        # ════════════════════════════════════════════════════════════════════
        for tbl_idx, tbl_el in enumerate(table_texts):
            tbl_md = _extract_table_markdown(tbl_el.text)
            tbl_summary = tbl_el.text[:512] if not tbl_md else tbl_md

            parent_id = str(uuid.uuid4())

            table_image_path_val = ""
            vision_description_val = ""

            if TABLE_RENDERER_AVAILABLE and tbl_el.raw_html:
                ticker = file_meta.get("ticker", "unknown")
                year = file_meta.get("year", "unknown")
                form = file_meta.get("filing_type", "unknown").replace("-", "")
                section_slug = anchor_id if anchor_id else "unknown"
                img_name = f"{ticker}_{year}_{form}_{section_slug}_table{tbl_idx+1}.png"
                table_image_path_val = render_table_to_image(
                    tbl_el.raw_html,
                    _TABLE_IMAGES_DIR,
                    img_name,
                    width=1200,
                )

            if VISION_EXTRACTOR_AVAILABLE and table_image_path_val:
                vision_description_val = describe_table_image(table_image_path_val)

            chunks.append(Chunk(
                text=tbl_summary,
                parent_text=tbl_el.text[:2000],
                table_markdown=tbl_md,
                table_image_path=table_image_path_val,
                vision_description=vision_description_val,
                company=_ticker_to_company(file_meta.get("ticker", "")),
                ticker=file_meta.get("ticker", ""),
                cik=file_meta.get("cik", ""),
                year=file_meta.get("year", 0),
                quarter=file_meta.get("quarter"),
                filing_type=file_meta.get("filing_type", ""),
                accession_number=file_meta.get("accession_number", ""),
                section=section,
                anchor_id=anchor_id,
                element_type="Table",
                chunk_index=chunk_idx,
                parent_chunk_id=parent_id,
                source_url=file_meta.get("source_url") or build_source_url(
                    file_meta.get("ticker", ""),
                    file_meta.get("cik", ""),
                ),
                htm_filename=file_meta.get("htm_filename", ""),
            ))
            chunk_idx += 1

    return chunks


def _ticker_to_company(ticker: str) -> str:
    """Map stock ticker symbol to full company name.
    
    Args:
        ticker: Stock ticker symbol (e.g. "NVDA")
        
    Returns:
        Full company name, or empty string if ticker not found
    """
    # Mapping of 10 major tech/finance companies
    names = {
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices, Inc.",
        "INTC": "Intel Corporation",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
        "AMZN": "Amazon.com, Inc.",
        "TSLA": "Tesla, Inc.",
        "JPM": "JPMorgan Chase & Co.",
    }
    # Return company name if ticker found, empty string otherwise
    return names.get(ticker, "")
