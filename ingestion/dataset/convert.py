""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Dataset Conversion & Export                                               ║
    ║  Loads/saves chunks to JSONL. Placeholder for Hugging Face datasets.       ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convert between chunk formats (dict, JSONL, HuggingFace datasets) for
    downstream ML pipelines. Handles serialization/deserialization.

Key Functions:
    load_chunks(jsonl_path)      - Load JSONL into FilingChunk objects
    save_jsonl(chunks)           - Save chunks to JSONL (data/processed/)
    to_hf_dataset(chunks)        - Convert to HuggingFace dataset (TODO)

Data:
    DEFAULT_PROCESSED_DIR        - Default output directory: data/processed/

Usage:
    chunks = load_chunks("NVDA_2024_10K_chunks.jsonl")
    path = save_jsonl(chunks, filename="merged_dataset.jsonl")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

from ingestion.dataset.schema import FilingChunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ══════════════════════════════════════════════════════════════════════════════
# Chunk Loading & Saving
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks(jsonl_path: str | Path) -> list[FilingChunk]:
    """Load chunks from JSONL file into FilingChunk objects.
    
    Reads one JSON object per line, deserializes to FilingChunk instances.
    
    Args:
        jsonl_path: Path to JSONL file with chunk objects
        
    Returns:
        List of FilingChunk objects
        
    Example:
        chunks = load_chunks("data/processed/NVDA_2024_10K_chunks.jsonl")
        # Returns list of ~925 FilingChunk objects
    """

    chunks: list[FilingChunk] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(FilingChunk.from_dict(json.loads(line)))
    return chunks


def save_jsonl(chunks: Iterable[FilingChunk], *, filename: str = "dataset_chunks.jsonl", processed_dir: str | Path | None = None) -> Path:
    """Save chunks as JSONL file in data/processed/ directory.
    
    One JSON object per line. Default directory is project root/data/processed/.
    
    Args:
        chunks: Iterable of FilingChunk objects
        filename: Output filename (default: dataset_chunks.jsonl)
        processed_dir: Output directory (default: data/processed/)
        
    Returns:
        Path to created JSONL file
        
    Example:
        path = save_jsonl(chunks, filename="all_filings.jsonl")
        # Writes to: /project/data/processed/all_filings.jsonl
    """
    processed_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    return out_path


def to_hf_dataset(chunks: Iterable[FilingChunk]):
    """Convert chunks into a Hugging Face dataset.
    
    TODO: Implement integration with huggingface_hub for dataset upload.
    Will support features like data push to HF Hub, streaming, etc.
    
    Args:
        chunks: Iterable of FilingChunk objects
        
    Raises:
        NotImplementedError: Not yet implemented
    """
    raise NotImplementedError("HF dataset conversion not implemented yet")
