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
import httpx
import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {
            "qdrant": {
                "host": config.QDRANT_HOST,
                "port": config.QDRANT_PORT,
                "collection": config.QDRANT_COLLECTION,
                "dense_dim": config.NEO4J_VECTOR_DIMENSIONS,
            },
            "models": {
                "dense_embedding": "BAAI/bge-small-en-v1.5",
                "sparse_embedding": "prithvida/Splade_PP_en_v1",
            },
        }

    with open(path) as f:
        return yaml.safe_load(f)


def get_client(cfg: dict | None = None) -> QdrantClient:
    if cfg is None:
        cfg = load_config()
    return QdrantClient(cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])


def create_collection(client: QdrantClient, cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    col = cfg["qdrant"]["collection"]
    dim = cfg["qdrant"]["dense_dim"]

    existing = [c.name for c in client.get_collections().collections]
    if col in existing:
        client.delete_collection(col)

    client.create_collection(
        collection_name=col,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def _embed_dense_remote(texts: list[str], base_url: str, model: str, batch_size: int = 64) -> list[list[float]]:
    results = []
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = client.post(
                f"{base_url.rstrip('/')}/v1/embeddings",
                json={"input": batch, "model": model},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            for item in data:
                results.append(item["embedding"])
    return results


def _embed_sparse_remote(texts: list[str], base_url: str, batch_size: int = 64) -> list[dict]:
    results = []
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = client.post(
                f"{base_url.rstrip('/')}/v1/sparse_embeddings",
                json={"input": batch},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            for item in data:
                results.append({
                    "indices": item["indices"],
                    "values": item["values"],
                })
    return results


def _load_dense_model(cfg: dict):
    base_url = getattr(config, "EMBEDDING_BASE_URL", "") or ""
    if base_url:
        return "remote", base_url, cfg["models"]["dense_embedding"]
    from fastembed import TextEmbedding
    return "local", TextEmbedding(cfg["models"]["dense_embedding"]), None


def _load_sparse_model(cfg: dict):
    base_url = getattr(config, "EMBEDDING_BASE_URL", "") or ""
    if base_url:
        return "remote", base_url
    from fastembed import SparseTextEmbedding
    return "local", SparseTextEmbedding(cfg["models"]["sparse_embedding"])


def embed_chunks(
    chunks: list[dict],
    dense_model,
    sparse_model,
    batch_size: int = 64,
) -> list[PointStruct]:
    texts = []
    for c in chunks:
        if c.get("element_type") == "Table" and c.get("vision_description"):
            tbl_md = c.get("table_markdown", "")
            texts.append(f"{c['vision_description']}\n\n{tbl_md}")
        else:
            texts.append(c["text"])
    points = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_chunks = chunks[start:end]

        print(f"  Embedding {start+1}-{end}/{total}...")

        if dense_model[0] == "remote":
            _, base_url, model_name = dense_model
            dense_vecs = _embed_dense_remote(batch_texts, base_url, model_name, batch_size=batch_size)
        else:
            _, local_model, _ = dense_model
            dense_vecs = [v.tolist() for v in local_model.embed(batch_texts)]

        if sparse_model[0] == "remote":
            _, base_url = sparse_model
            sparse_vecs = _embed_sparse_remote(batch_texts, base_url, batch_size=batch_size)
        else:
            _, local_sparse = sparse_model
            sparse_vecs = list(local_sparse.embed(batch_texts))

        for i, chunk in enumerate(batch_chunks):
            dv = dense_vecs[i] if isinstance(dense_vecs[i], list) else dense_vecs[i]

            if isinstance(sparse_vecs[i], dict):
                sv_indices = sparse_vecs[i]["indices"]
                sv_values = sparse_vecs[i]["values"]
            else:
                sv_indices = sparse_vecs[i].indices.tolist()
                sv_values = sparse_vecs[i].values.tolist()

            embed_text = chunk.get("text", "")
            if chunk.get("element_type") == "Table" and chunk.get("vision_description"):
                tbl_md = chunk.get("table_markdown", "")
                embed_text = f"{chunk['vision_description']}\n\n{tbl_md}"

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
                            indices=sv_indices,
                            values=sv_values,
                        ),
                    },
                    payload=payload,
                )
            )

    return points


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    cfg: dict | None = None,
    batch_size: int = 200,
) -> None:
    if cfg is None:
        cfg = load_config()
    col = cfg["qdrant"]["collection"]

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=col, points=batch)
        print(f"  Upserted {start+1}-{min(start+batch_size, len(points))}/{len(points)}")


def ingest_jsonl(jsonl_path: str | Path, cfg: dict | None = None) -> int:
    if cfg is None:
        cfg = load_config()

    client = get_client(cfg)
    dense_model = _load_dense_model(cfg)
    sparse_model = _load_sparse_model(cfg)

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
    points = embed_chunks(chunks, dense_model, sparse_model)
    upsert_points(client, points, cfg)
    return len(points)


def _get_ingested_signatures(client: QdrantClient, cfg: dict) -> set[str]:
    col = cfg["qdrant"]["collection"]
    signatures = set()
    offset = None
    while True:
        results, offset = client.scroll(col, limit=500, offset=offset, with_payload=True)
        if not results:
            break
        for r in results:
            p = r.payload
            sig = f"{p.get('ticker','')}_{p.get('filing_type','')}_{p.get('year','')}_{p.get('quarter','')}"
            signatures.add(sig)
        if offset is None:
            break
    return signatures


def _file_to_signature(filepath: Path) -> str:
    stem = filepath.stem.replace("_chunks", "").replace("_clean", "")
    parts = stem.split("_")
    ticker = parts[0] if parts else ""
    filing_type = ""
    year = ""
    quarter = ""
    for p in parts[1:]:
        if p in ("10K", "10-K"):
            filing_type = "10-K"
        elif p in ("10Q", "10-Q"):
            filing_type = "10-Q"
        elif len(p) == 4 and p.isdigit():
            year = p
        elif p.startswith("Q") and len(p) == 2 and p[1].isdigit():
            quarter = p[1]
    return f"{ticker}_{filing_type}_{year}_{quarter}"


def ingest_all(
    recreate: bool = False,
    resume: bool = False,
) -> int:
    cfg = load_config()
    client = get_client(cfg)

    if recreate:
        create_collection(client, cfg)

    ingested_sigs = set()
    if resume and not recreate:
        ingested_sigs = _get_ingested_signatures(client, cfg)
        print(f"Resume mode: {len(ingested_sigs)} filings already in Qdrant")

    processed_dir = PROJECT_ROOT / config.INGESTION_PROCESSED_DIR
    if not processed_dir.exists():
        print(f"Directory {processed_dir} not found")
        return 0

    total = 0
    jsonl_files = sorted(processed_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No *_chunks.jsonl files found")
        return 0

    print(f"Ingesting {len(jsonl_files)} filing(s) into '{cfg['qdrant']['collection']}'...\n")

    dense_model = _load_dense_model(cfg)
    sparse_model = _load_sparse_model(cfg)

    for jsonl_file in jsonl_files:
        sig = _file_to_signature(jsonl_file)
        if sig in ingested_sigs:
            print(f"Skipping {jsonl_file.name} (already ingested)")
            continue

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
        upsert_points(client, points, cfg)
        total += len(points)
        print()

    print(f"Done. {total} points ingested into '{cfg['qdrant']['collection']}'.")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chunk JSONL files into Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--resume", action="store_true", help="Skip filings already in Qdrant")
    args = parser.parse_args()

    ingest_all(recreate=args.recreate, resume=args.resume)
