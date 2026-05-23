""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Chunking API Wrapper                                                      ║
    ║  Convenience re-export for ingestion/core/chunker API.                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Top-level API for the chunking module. Re-exports Chunk dataclass and
    chunk_filing() function for direct use without full import path.

Usage:
    # Instead of: from ingestion.core.chunker import chunk_filing, Chunk
    # Use this shorter path:
    from ingestion.chunking import chunk_filing, Chunk
    
    chunks = chunk_filing("file_clean.htm", "file.htm")
    # chunks is list[Chunk]

Exports:
    - Chunk: RAG-ready chunk dataclass with metadata
    - chunk_filing: Main orchestrator function
"""

from ingestion.core.chunker import Chunk, chunk_filing  # re-export for convenience

__all__ = ["Chunk", "chunk_filing"]
