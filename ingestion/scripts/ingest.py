""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Qdrant Ingest Pipeline                                                    ║
    ║  Embeds chunks with dense + sparse vectors, upserts to Qdrant vector DB.  ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convert JSONL chunks into dense + sparse vectors and ingest into Qdrant
    vector database for hybrid search capabilities.

Run:
    # Ingest all JSONL files in data/processed/ into Qdrant
    python -m ingestion.scripts.ingest
    
    # Recreate collection (delete old data)
    python -m ingestion.scripts.ingest --recreate

Configuration:
    Source: config.yaml in project root (or uses defaults)
    Default Dense Model: BAAI/bge-small-en-v1.5 (384 dims)
    Default Sparse Model: prithvida/Splade_PP_en_v1
    Qdrant Host: localhost:6333 (default)

Models:
    Dense Vectors:  384-dim semantic embeddings (cosine distance)
    Sparse Vectors: BM25-style term frequency (IDF weighting)
    
    Hybrid approach enables both semantic and keyword search.
"""

import json
import uuid
from pathlib import Path
import sys

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
    SparseVector,
)
from fastembed import TextEmbedding, SparseTextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Management
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load Qdrant and model config from YAML file.
    
    Falls back to reasonable defaults if file doesn't exist.
    
    Args:
        path: Path to config.yaml (default: project root)
        
    Returns:
        Config dict with qdrant connection and model settings
    """
    path = Path(path)
    if not path.exists():
        # Return sensible defaults
        return {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection": "prism_filings",
                "dense_dim": 384,
            },
            "models": {
                "dense_embedding": "BAAI/bge-small-en-v1.5",
                "sparse_embedding": "prithvida/Splade_PP_en_v1",
            },
        }

    with open(path) as f:
        return yaml.safe_load(f)


def get_client(config: dict | None = None) -> QdrantClient:
    """Create Qdrant client from config.
    
    Args:
        config: Config dict (loads from file if None)
        
    Returns:
        Qdrant client instance
    """
    if config is None:
        config = load_config()
    return QdrantClient(config["qdrant"]["host"], port=config["qdrant"]["port"])


def create_collection(client: QdrantClient, config: dict | None = None) -> None:
    """Create or recreate collection with dense + sparse vectors.
    
    Deletes existing collection if present, then creates new one.
    
    Args:
        client: Qdrant client
        config: Config dict (loads from file if None)
    """
    if config is None:
        config = load_config()
    col = config["qdrant"]["collection"]
    dim = config["qdrant"]["dense_dim"]

    # Delete existing collection
    existing = [c.name for c in client.get_collections().collections]
    if col in existing:
        client.delete_collection(col)

    # Create collection with dense + sparse vectors
    client.create_collection(
        collection_name=col,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def _load_dense_model(config: dict) -> TextEmbedding:
    """Load dense embedding model (semantic embeddings).
    
    Args:
        config: Config dict
        
    Returns:
        TextEmbedding model instance
    """
    return TextEmbedding(config["models"]["dense_embedding"])


def _load_sparse_model(config: dict) -> SparseTextEmbedding:
    """Load sparse embedding model (term frequency).
    
    Args:
        config: Config dict
        
    Returns:
        SparseTextEmbedding model instance
    """
    return SparseTextEmbedding(config["models"]["sparse_embedding"])


# ══════════════════════════════════════════════════════════════════════════════
# Embedding and Ingestion
# ══════════════════════════════════════════════════════════════════════════════

def embed_chunks(
    chunks: list[dict],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    batch_size: int = 64,
) -> list[PointStruct]:
    """Convert chunks to dense + sparse vectors as Qdrant PointStruct.
    
    Args:
        chunks: List of chunk dicts from JSONL
        dense_model: Dense embedding model
        sparse_model: Sparse embedding model
        batch_size: Batch size for embedding (default 64)
        
    Returns:
        List of PointStruct objects ready for Qdrant upsert
    """
    texts = [c["text"] for c in chunks]
    points = []
    total = len(texts)

    # Batch embedding for efficiency
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_chunks = chunks[start:end]

        print(f"  Embedding {start+1}-{end}/{total}...")

        # Generate dense and sparse vectors
        dense_vecs = list(dense_model.embed(batch_texts))
        sparse_vecs = list(sparse_model.embed(batch_texts))

        # Create PointStruct for each chunk
        for i, chunk in enumerate(batch_chunks):
            dv = dense_vecs[i].tolist()
            sv = sparse_vecs[i]

            # Prepare payload (all chunk metadata except chunk_id)
            payload = {k: v for k, v in chunk.items() if k not in ("chunk_id",)}
            for key, val in payload.items():
                if isinstance(val, Path):
                    payload[key] = str(val)

            points.append(
                PointStruct(
                    id=chunk.get("chunk_id") or str(uuid.uuid4()),
                    vector={
                        "dense": dv,
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

    return points


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    config: dict | None = None,
    batch_size: int = 100,
) -> None:
    """Upsert points into Qdrant collection in batches.
    
    Args:
        client: Qdrant client
        points: List of PointStruct to upsert
        config: Config dict (loads from file if None)
        batch_size: Batch size for upsert (default 100)
    """
    if config is None:
        config = load_config()
    col = config["qdrant"]["collection"]

    # Batch upsert for efficiency
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=col, points=batch)
        print(f"  Upserted {start+1}-{min(start+batch_size, len(points))}/{len(points)}")


def ingest_jsonl(jsonl_path: str | Path, config: dict | None = None) -> int:
    """Ingest a single JSONL file of chunks into Qdrant.
    
    Args:
        jsonl_path: Path to _chunks.jsonl file
        config: Config dict (loads from file if None)
        
    Returns:
        Number of points ingested
    """
    if config is None:
        config = load_config()

    client = get_client(config)
    dense_model = _load_dense_model(config)
    sparse_model = _load_sparse_model(config)

    # Load chunks from JSONL
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        print(f"  No chunks in {jsonl_path}")
        return 0

    print(f"  {Path(jsonl_path).name}: {len(chunks)} chunks")
    # Embed and upsert
    points = embed_chunks(chunks, dense_model, sparse_model)
    upsert_points(client, points, config)
    return len(points)


def ingest_all(
    processed_dir: str | Path = "data/processed",
    config_path: str | Path = CONFIG_PATH,
    recreate: bool = False,
) -> int:
    """Ingest all JSONL files from processed directory into Qdrant.
    
    Args:
        processed_dir: Directory with *_chunks.jsonl files
        config_path: Path to config.yaml
        recreate: If True, delete and recreate collection
        
    Returns:
        Total number of points ingested
    """
    config = load_config(config_path)
    client = get_client(config)

    # Optionally recreate collection (clears old data)
    if recreate:
        create_collection(client, config)

    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(f"Directory {processed_dir} not found")
        return 0

    total = 0
    jsonl_files = sorted(processed_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No *_chunks.jsonl files found")
        return 0

    print(f"Ingesting {len(jsonl_files)} filing(s) into '{config['qdrant']['collection']}'...\n")

    # Reuse models for all files (more efficient)
    dense_model = _load_dense_model(config)
    sparse_model = _load_sparse_model(config)

    for jsonl_file in jsonl_files:
        chunks = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            continue

        print(f"Processing {jsonl_file.name}: {len(chunks)} chunks")
        points = embed_chunks(chunks, dense_model, sparse_model)
        upsert_points(client, points, config)
        total += len(points)
        print()

    print(f"Done. {total} points ingested into '{config['qdrant']['collection']}'.")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chunk JSONL files into Qdrant")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--file", type=str, help="Ingest a single JSONL file instead of all")
    args = parser.parse_args()

    if args.file:
        count = ingest_jsonl(args.file, load_config(args.config))
    else:
        count = ingest_all(args.processed_dir, args.config, args.recreate)
