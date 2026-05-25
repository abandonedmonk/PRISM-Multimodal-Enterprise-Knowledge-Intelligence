""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Graph Retriever                                                            ║
    ║  Neo4j-backed local and global search for GraphRAG.                       ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Implements two search modes from GraphRAG:
    - Local Search: entity-focused questions ("How is X related to Y?")
    - Global Search: thematic questions ("What are major trends in the corpus?")

Usage:
    from retrieval.graph_retriever import GraphRetriever

    retriever = GraphRetriever()
    local_context = retriever.local_search(query, top_k=5)
    global_context = retriever.global_search(query, top_k=5)
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None

import os


def _get_driver():
    # Import the neo4j driver at runtime to avoid hard dependency during
    # import time for unrelated commands or tests.
    try:
        neo4j = importlib.import_module("neo4j")
    except Exception as exc:
        raise ImportError("neo4j is required for GraphRetriever") from exc

    # Prefer values from the central `config` module, fall back to env vars.
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

    # Return the driver; callers should close it when finished.
    return neo4j.GraphDatabase.driver(uri, auth=(user, pwd))


class GraphRetriever:
    """High-level helper for Neo4j-backed retrieval used by PRISM.

    The class is intentionally small: it focuses on assembling context
    strings and identifiers to feed into LLM prompts or the hybrid
    retriever. Methods use simple, well-defined return shapes so callers
    can easily consume results.
    """

    def __init__(self):
        # Initialize a cached driver instance. Errors will bubble up if
        # the neo4j dependency or server is unavailable.
        self.driver = _get_driver()

        # Attempt to create any required vector index (best-effort).
        self._ensure_vector_index()

    def _ensure_vector_index(self):
        # Best-effort creation of a Neo4j vector index. Failures are
        # non-fatal because the index might already exist or the server may
        # not support this feature in older releases.
        try:
            import config as cfg

            index_name = getattr(cfg, "NEO4J_VECTOR_INDEX_NAME", "entity_embeddings")
            dims = getattr(cfg, "NEO4J_VECTOR_DIMENSIONS", 384)

            with self.driver.session() as session:
                result = session.run(f"SHOW INDEXES YIELD name WHERE name = '{index_name}'")
                if not result.single():
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (e:Entity) ON (e.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {dims},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
        except Exception:
            # Ignore problems here; the calling code should handle cases
            # where the index is required but not present.
            pass

    def local_search(
        self,
        entity_keys: list[str] | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        hop: int = 2,
    ) -> dict:
        """Local search: traverse graph from seed entities to build context.

        Args:
            entity_keys: List of seed entity keys to start from
            query_embedding: If provided, find similar entities by vector search
            top_k: Number of top entities to consider per search mode
            hop: Number of graph hops to traverse (default 2)

        Returns:
            Dict with:
            {
                "entities": [...],   # Entity dicts with summaries
                "relations": [...],  # Relation dicts in the neighborhood
                "context_text": "...",  # Assembled text for LLM
                "chunk_ids": [...],  # All associated chunk IDs
            }
        """
        # Build a context around seed entities or an embedding query. The
        # return value includes entities, relations, a concatenated
        # `context_text` suitable for LLM prompts, and `chunk_ids` for
        # fetching source documents.
        with self.driver.session() as session:
            if query_embedding:
                index_name = (
                    (config.NEO4J_VECTOR_INDEX_NAME if config else None)
                    or os.getenv("NEO4J_VECTOR_INDEX_NAME", "entity_embeddings")
                )
                result = session.run(f"""
                    CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.key AS key, node.name AS name,
                           node.type AS type, node.summary AS summary,
                           score
                """, top_k=top_k, embedding=query_embedding)
                seed_records = [dict(r) for r in result]
                if seed_records:
                    entity_keys = [r["key"] for r in seed_records]

            if not entity_keys:
                # No seeds available — return empty structured response.
                return {
                    "entities": [],
                    "relations": [],
                    "context_text": "",
                    "chunk_ids": [],
                }

            placeholders = ", ".join([f"$key{i}" for i in range(len(entity_keys))])
            params = {f"key{i}": k for i, k in enumerate(entity_keys)}

            # Find neighbor relationships up to `hop` hops away and pull
            # relation metadata. We limit the returned set for performance.
            neighbors_result = session.run(f"""
                MATCH (e:Entity)
                WHERE e.key IN [{placeholders}]
                CALL (e) {{
                    WITH e
                    MATCH path = (e)-[r:RELATES*1..{hop}]-(neighbor:Entity)
                    WITH e, neighbor, r, relationships(path) AS rels
                    UNWIND rels AS rel
                    WITH e, neighbor, rel
                    RETURN e.key AS from_key, e.name AS from_name,
                           neighbor.key AS to_key, neighbor.name AS to_name,
                           rel.type AS relation_type,
                           rel.description AS description,
                           rel.weight AS weight
                    LIMIT 100
                }}
                RETURN from_key, from_name, to_key, to_name,
                       collect({{type: relation_type, description: description, weight: weight}}) AS relations
            """, **params)
            neighbor_data = [dict(r) for r in neighbors_result]

            # Expand the set of keys to fetch full entity details.
            entity_keys_set = set(entity_keys)
            for nd in neighbor_data:
                entity_keys_set.add(nd.get("from_key", ""))
                entity_keys_set.add(nd.get("to_key", ""))

            entity_details = {}
            if entity_keys_set:
                ep = ", ".join([f"$ek{i}" for i in range(len(entity_keys_set))])
                epp = {f"ek{i}": k for i, k in enumerate(entity_keys_set)}
                details_result = session.run(f"""
                    MATCH (e:Entity)
                    WHERE e.key IN [{ep}]
                    RETURN e.key AS key, e.name AS name, e.type AS type,
                           e.summary AS summary, e.occurrence_count AS count,
                           e.chunk_ids AS chunk_ids
                """, **epp)
                for r in details_result:
                    d = dict(r)
                    entity_details[d["key"]] = d

            # Normalize relations into a simple list for callers.
            relations_list = []
            for nd in neighbor_data:
                for rel in nd.get("relations", []):
                    relations_list.append({
                        "source": nd.get("from_name", ""),
                        "target": nd.get("to_name", ""),
                        "type": rel.get("type", ""),
                        "description": rel.get("description", ""),
                    })

            # Accumulate unique chunk ids referenced by matching entities.
            chunk_ids_set = set()
            for ed in entity_details.values():
                for cid in (ed.get("chunk_ids") or []):
                    chunk_ids_set.add(cid)

            # Build a human-readable context composed of entity summaries
            # and relation descriptions (used for prompt construction).
            context_parts = []
            for key, ed in entity_details.items():
                if ed.get("summary"):
                    context_parts.append(
                        f"[{ed['type']}] {ed['name']}: {ed['summary']}"
                    )

            # Add up to 30 relation descriptions to keep context compact.
            for rel in relations_list[:30]:
                if rel.get("description"):
                    context_parts.append(
                        f"{rel['source']} --[{rel['type']}]--> {rel['target']}: {rel['description']}"
                    )

            return {
                "entities": list(entity_details.values()),
                "relations": relations_list,
                "context_text": "\n\n".join(context_parts),
                "chunk_ids": list(chunk_ids_set),
            }

    def global_search(
        self,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> dict:
        """Global search: find relevant community reports for thematic queries.

        Falls back to entity summaries from community members if reports are missing.

        Args:
            query_embedding: Embedding of the user query
            top_k: Number of top community reports to retrieve

        Returns:
            Dict with:
            {
                "communities": [...],   # Community report dicts or entity summaries
                "context_text": "...",  # Assembled text
                "chunk_ids": [...],     # All associated chunk IDs
            }
        """
        if not query_embedding:
            return {"communities": [], "context_text": "", "chunk_ids": []}

        with self.driver.session() as session:
            index_name = (
                (config.NEO4J_VECTOR_INDEX_NAME if config else None)
                or os.getenv("NEO4J_VECTOR_INDEX_NAME", "entity_embeddings")
            )

            result = session.run("""
                MATCH (c:Community)
                WHERE c.title IS NOT NULL AND c.summary IS NOT NULL
                RETURN c.community_id AS community_id,
                       c.title AS title,
                       c.summary AS summary,
                       c.key_points AS key_points,
                       c.risks AS risks,
                       c.member_count AS size
                ORDER BY size DESC
                LIMIT $top_k
            """, top_k=top_k * 2)

            all_communities = [dict(r) for r in result]

            if not all_communities:
                fallback_result = session.run("""
                    MATCH (c:Community)
                    RETURN c.community_id AS community_id,
                           c.member_count AS size,
                           c.level AS level
                    ORDER BY size DESC
                    LIMIT $top_k
                """, top_k=top_k)
                communities_fallback = [dict(r) for r in fallback_result]

                context_parts = []
                chunk_ids_set = set()

                for comm in communities_fallback[:top_k]:
                    members_result = session.run("""
                        MATCH (c:Community {community_id: $cid})-[:HAS_MEMBER]->(e:Entity)
                        WHERE e.summary IS NOT NULL
                        RETURN e.name AS name, e.type AS type, e.summary AS summary
                        LIMIT 10
                    """, cid=comm.get("community_id"))

                    member_summaries = []
                    for mr in members_result:
                        d = dict(mr)
                        if d.get("summary"):
                            member_summaries.append(f"[{d['type']}] {d['name']}: {d['summary']}")

                    if member_summaries:
                        context_parts.append(
                            f"[Community {comm.get('community_id')} - {comm.get('size')} entities]\n"
                            + "\n".join(member_summaries)
                        )

                    chunk_result = session.run("""
                        MATCH (c:Community {community_id: $cid})-[:HAS_MEMBER]->(e:Entity)
                        RETURN collect(e.chunk_ids) AS all_chunk_ids
                    """, cid=comm.get("community_id"))
                    for r in chunk_result:
                        for cid_list in (r["all_chunk_ids"] or []):
                            if isinstance(cid_list, list):
                                for cid in cid_list:
                                    chunk_ids_set.add(cid)

                return {
                    "communities": communities_fallback[:top_k],
                    "context_text": "\n\n---\n\n".join(context_parts),
                    "chunk_ids": list(chunk_ids_set),
                }

            context_parts = []
            chunk_ids_set = set()

            for comm in all_communities[:top_k]:
                parts = [f"[Community] {comm.get('title', '')}"]
                if comm.get("summary"):
                    parts.append(f"Summary: {comm['summary']}")
                if comm.get("key_points"):
                    pts = comm["key_points"]
                    if isinstance(pts, list):
                        parts.append("Key Points: " + "; ".join(pts))
                if comm.get("risks"):
                    risks = comm["risks"]
                    if isinstance(risks, list):
                        parts.append("Risks: " + "; ".join(risks))

                context_parts.append("\n".join(parts))

                members_result = session.run("""
                    MATCH (c:Community {community_id: $cid})-[:HAS_MEMBER]->(e:Entity)
                    RETURN collect(e.chunk_ids) AS all_chunk_ids
                """, cid=comm.get("community_id"))
                for r in members_result:
                    for cid_list in (r["all_chunk_ids"] or []):
                        if isinstance(cid_list, list):
                            for cid in cid_list:
                                chunk_ids_set.add(cid)

            return {
                "communities": all_communities[:top_k],
                "context_text": "\n\n---\n\n".join(context_parts),
                "chunk_ids": list(chunk_ids_set),
            }

    def get_entity_info(self, entity_key: str) -> dict | None:
        """Retrieve stored properties for a single entity node.

        Returns a dict with keys `key, name, type, summary, count, chunk_ids,
        community_id` or `None` if the entity is not present.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {key: $key})
                RETURN e.key AS key, e.name AS name, e.type AS type,
                       e.summary AS summary, e.occurrence_count AS count,
                       e.chunk_ids AS chunk_ids, e.community_id AS community_id
            """, key=entity_key)
            record = result.single()
            return dict(record) if record else None

    def resolve_entity_key(self, name: str, entity_type: str | None = None) -> str | None:
        """Resolve a canonical entity key from a human name.

        Returns the most frequent matching entity key or None if not found.
        """
        if not name:
            return None

        with self.driver.session() as session:
            if entity_type:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($name)
                      AND toLower(e.type) = toLower($etype)
                    RETURN e.key AS key, e.occurrence_count AS count
                    ORDER BY count DESC
                    LIMIT 1
                """, name=name, etype=entity_type)
            else:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($name)
                    RETURN e.key AS key, e.occurrence_count AS count
                    ORDER BY count DESC
                    LIMIT 1
                """, name=name)

            record = result.single()
            return record.get("key") if record else None

    def get_communities(self) -> list[dict]:
        """Return a list of community summaries (id, title, summary, size).

        The list is ordered by community size descending.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Community)
                RETURN c.community_id AS community_id,
                       c.title AS title, c.summary AS summary,
                       c.member_count AS size, c.level AS level
                ORDER BY size DESC
            """)
            return [dict(r) for r in result]

    def close(self):
        self.driver.close()
