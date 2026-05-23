""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Filing Chunk Schema                                                       ║
    ║  Dataclass defining chunk structure and metadata fields for datasets.      ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Define standardized schema for filing chunks to enable serialization,
    deserialization, and type hints across the ingestion pipeline.

Key Classes:
    FilingChunk                  - Complete chunk with 18 metadata fields

Usage:
    chunk = FilingChunk.from_dict(json_obj)
    # Automatically validates field types and deserializes JSONL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FilingChunk:
    """RAG-ready chunk with complete filing metadata.
    
    This schema defines the standard format for serialized chunks in JSONL files.
    Matches ingestion.core.chunker.Chunk for compatibility.
    
    Attributes:
        text: Dense chunk text for embedding (1024 tokens)
        parent_text: Parent context (3500 tokens)
        table_markdown: Markdown table (if Table element)
        company: Company name
        ticker: Stock ticker
        cik: SEC identifier
        year: Fiscal year
        quarter: Fiscal quarter (None for 10-K)
        filing_type: Form type (10-K or 10-Q)
        accession_number: SEC accession ID
        section: Section label (MD&A, etc.)
        anchor_id: Section ID (item7, etc.)
        element_type: NarrativeText or Table
        chunk_index: Position in all chunks
        parent_chunk_id: Link to parent chunk
        chunk_id: Unique identifier
        source_url: SEC EDGAR URL
        htm_filename: Original filename
    """
    text: str
    parent_text: str
    table_markdown: str
    company: str
    ticker: str
    cik: str
    year: int
    quarter: int | None
    filing_type: str
    accession_number: str
    section: str
    anchor_id: str
    element_type: str
    chunk_index: int
    parent_chunk_id: str
    chunk_id: str
    source_url: str
    htm_filename: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilingChunk":
        """Deserialize chunk from dict (e.g., from JSON).
        
        Args:
            data: Dictionary with chunk fields
            
        Returns:
            FilingChunk instance
        """
        return cls(**data)
