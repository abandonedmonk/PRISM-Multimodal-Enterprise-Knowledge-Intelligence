""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Neo4j Vector Indexer                                                        ║
    ║  Embeds entities and community reports, indexes them in Neo4j vector index. ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    After building the Neo4j graph and generating community reports, embed
    entity summaries and community report texts into Neo4j's vector index
    for ANN-based retrieval in local and global search.

Usage:
    from ingestion.core.vector_indexer import index_entities_and_communities

    count = index_entities_and_communities()
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None

import os
import json

import httpx


def _get_driver():
    try:
        neo4j = importlib.import_module("neo4j")

        uri = (
            (config.NEO4J_URI if config else "")
            or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        user = (
            (config.NEO4J_USER if config else "")
            or os.getenv("NEO4J_USER", "neo4j")
        )
        pwd = (
            (config.NEO4J_PASSWORD if config else "")
            or os.getenv("NEO4J_PASSWORD", "password")
        )
        return neo4j.GraphDatabase.driver(uri, auth=(user, pwd))
    except Exception:
        return None


def _embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    base_url = (
        (getattr(config, "EMBEDDING_BASE_URL", "") if config else "")
        or os.getenv("EMBEDDING_BASE_URL", "")
    )
    model_name = (
        (getattr(config, "EMBEDDING_MODEL", "") if config else "")
        or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    if base_url:
        all_embeddings = []
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                resp = client.post(
                    f"{base_url.rstrip('/')}/v1/embeddings",
                    json={"input": batch, "model": model_name},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d["index"])
                for item in data:
                    all_embeddings.append(item["embedding"])
        return all_embeddings

    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name)
        return [v.tolist() for v in model.embed(texts)]
    except Exception:
        return [[0.0] * 384 for _ in texts]


def index_entities_and_communities() -> int:
    """Embed and index all entities and community reports in Neo4j vector index.

    Returns:
        Number of items indexed
    """
    driver = _get_driver()
    if driver is None:
        print("  Neo4j driver not available")
        return 0

    indexed = 0

    index_name = (
        (config.NEO4J_VECTOR_INDEX_NAME if config else None)
        or os.getenv("NEO4J_VECTOR_INDEX_NAME", "entity_embeddings")
    )
    dims = (
        (config.NEO4J_VECTOR_DIMENSIONS if config else None)
        or int(os.getenv("NEO4J_VECTOR_DIMENSIONS", "384"))
    )

    with driver.session() as session:
        try:
            session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
        except Exception as e:
            print(f"  Vector index creation note: {e}")

    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.summary IS NOT NULL AND e.summary <> ''
            RETURN e.key AS key, e.name AS name,
                   e.type AS type, e.summary AS summary
        """)
        entities = [dict(r) for r in result]

    print(f"  Embedding {len(entities)} entities (batched)...")
    entity_texts = [
        f"{ent['name']}\nType: {ent['type']}\nSummary: {ent['summary']}"
        for ent in entities
    ]
    entity_embeddings = _embed_texts(entity_texts)

    for ent, embedding in zip(entities, entity_embeddings):
        if embedding and any(v != 0 for v in embedding):
            with driver.session() as session:
                session.run("""
                    MATCH (e:Entity {key: $key})
                    SET e.embedding = $embedding
                """, key=ent["key"], embedding=embedding)
            indexed += 1

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Community)
            WHERE c.title IS NOT NULL AND c.summary IS NOT NULL
            RETURN c.community_id AS community_id,
                   c.title AS title, c.summary AS summary,
                   c.key_points AS key_points
        """)
        communities = [dict(r) for r in result]

    print(f"  Embedding {len(communities)} community reports (batched)...")
    community_texts = []
    for comm in communities:
        parts = [f"{comm['title']}", f"{comm['summary']}"]
        if comm.get("key_points"):
            pts = comm["key_points"]
            if isinstance(pts, list):
                parts.append("Key points: " + "; ".join(pts))
        community_texts.append("\n".join(parts))

    community_embeddings = _embed_texts(community_texts)

    for comm, embedding in zip(communities, community_embeddings):
        if embedding and any(v != 0 for v in embedding):
            with driver.session() as session:
                session.run("""
                    MATCH (c:Community {community_id: $cid})
                    SET c.embedding = $embedding
                """, cid=comm["community_id"], embedding=embedding)
            indexed += 1

    driver.close()
    print(f"  Indexed {indexed} items total")
    return indexed