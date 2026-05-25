""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Neo4j Graph Builder                                                        ║
    ║  Constructs knowledge graph from LLM-extracted entities and relations.     ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Build and persist a Neo4j knowledge graph from entity extraction results.
    Handles deduplication, evidence aggregation, and idempotent MERGE operations.

Usage:
    from ingestion.core.graph_builder import build_graph

    build_graph(extractions)  # List from entity_extractor.extract_from_chunks()

Schema:
    Nodes: (e:Entity {key, name, type, summary, occurrence_count, chunk_ids, ...})
    Edges: (s)-[r:RELATES {type, weight, strength_evidence, source_chunks}]->(t)
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

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "password"

import os


def _get_driver():
    try:
        neo4j = importlib.import_module("neo4j")
    except Exception as exc:
        raise ImportError("neo4j is required to build the graph") from exc

    uri = (
        (config.NEO4J_URI if config else "")
        or os.getenv("NEO4J_URI", DEFAULT_URI)
    )
    user = (
        (config.NEO4J_USER if config else "")
        or os.getenv("NEO4J_USER", DEFAULT_USER)
    )
    pwd = (
        (config.NEO4J_PASSWORD if config else "")
        or os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD)
    )
    return neo4j.GraphDatabase.driver(uri, auth=(user, pwd))


def _canonical_key(name: str, entity_type: str) -> str:
    return f"{name.lower().strip()}::{entity_type}"


def _get_or_create_constraints(driver):
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT entity_key IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.key IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT community_id IF NOT EXISTS
            FOR (c:Community) REQUIRE c.community_id IS UNIQUE
        """)


def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Graph cleared.")


def build_graph(extractions: list[dict], clear: bool = True):
    """Build Neo4j graph from entity extraction results.

    Args:
        extractions: List of extraction dicts from entity_extractor.extract_from_chunks()
        clear: If True, clear existing graph before building (default True)

    Example:
        from ingestion.core.entity_extractor import extract_from_chunks
        from ingestion.core.graph_builder import build_graph

        chunks = [{"chunk_id": "abc123", "text": "NVIDIA reported..."}]
        extractions = extract_from_chunks(chunks)
        build_graph(extractions)
    """
    driver = _get_driver()

    try:
        _get_or_create_constraints(driver)
    except Exception as e:
        print(f"  Constraint creation warning: {e}")

    if clear:
        clear_graph(driver)

    node_map: dict[str, dict] = {}
    edge_map: dict[str, dict] = {}

    for extraction in extractions:
        chunk_id = extraction.get("chunk_id", "")
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])
        chunk_entity_type: dict[str, str] = {}

        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "") or "Concept"
            key = _canonical_key(entity_name, entity_type)
            if not key or key == "::":
                continue
            chunk_entity_type[entity_name.lower().strip()] = entity_type

            if key not in node_map:
                node_map[key] = {
                    "name": entity_name,
                    "type": entity_type,
                    "description_evidence": [],
                    "occurrence_count": 0,
                    "chunk_ids": set(),
                }

            node_map[key]["occurrence_count"] += 1
            node_map[key]["chunk_ids"].add(chunk_id)
            desc = entity.get("description", "")
            if desc:
                node_map[key]["description_evidence"].append(desc)

        for relation in relations:
            src_name = relation.get("source", "")
            tgt_name = relation.get("target", "")
            src_norm = src_name.lower().strip()
            tgt_norm = tgt_name.lower().strip()

            src_type = chunk_entity_type.get(src_norm)
            tgt_type = chunk_entity_type.get(tgt_norm)

            if not src_type:
                for key, node in node_map.items():
                    if node.get("name", "").lower().strip() == src_norm:
                        src_type = node.get("type") or "Concept"
                        break
            if not tgt_type:
                for key, node in node_map.items():
                    if node.get("name", "").lower().strip() == tgt_norm:
                        tgt_type = node.get("type") or "Concept"
                        break

            src = _canonical_key(src_name, src_type or "Concept")
            tgt = _canonical_key(tgt_name, tgt_type or "Concept")
            rel_type = relation.get("relation", "RELATES")

            if src == "::" or tgt == "::":
                src_key = f"{src_norm}::Concept"
                tgt_key = f"{tgt_norm}::Concept"
                src = src_key
                tgt = tgt_key

            edge_key = f"{src}|{rel_type}|{tgt}"
            if edge_key not in edge_map:
                edge_map[edge_key] = {
                    "source": relation.get("source", ""),
                    "target": relation.get("target", ""),
                    "source_key": src,
                    "target_key": tgt,
                    "relation_type": rel_type,
                    "weight": 0,
                    "strength_evidence": [],
                    "descriptions": [],
                    "source_chunks": set(),
                }

            edge_map[edge_key]["weight"] += 1
            edge_map[edge_key]["source_chunks"].add(chunk_id)
            strength = relation.get("strength", 0.5)
            if strength:
                edge_map[edge_key]["strength_evidence"].append(float(strength))
            desc = relation.get("description", "")
            if desc:
                edge_map[edge_key]["descriptions"].append(desc)

    print(f"  Aggregated {len(node_map)} nodes, {len(edge_map)} edges")

    with driver.session() as session:
        for key, node in node_map.items():
            summary = max(node["description_evidence"], key=len) if node["description_evidence"] else node["name"]

            session.run("""
                MERGE (e:Entity {key: $key})
                SET e.name = $name,
                    e.type = $type,
                    e.summary = $summary,
                    e.occurrence_count = $occurrence_count,
                    e.chunk_ids = $chunk_ids
            """,
                key=key,
                name=node["name"],
                type=node["type"],
                summary=summary,
                occurrence_count=node["occurrence_count"],
                chunk_ids=list(node["chunk_ids"]),
            )

        for edge_key, edge in edge_map.items():
            avg_strength = (
                sum(edge["strength_evidence"]) / len(edge["strength_evidence"])
                if edge["strength_evidence"]
                else 0.5
            )
            rel_desc = max(edge["descriptions"], key=len) if edge["descriptions"] else edge["relation_type"]

            session.run("""
                MATCH (s:Entity {key: $source_key})
                MATCH (t:Entity {key: $target_key})
                MERGE (s)-[r:RELATES {type: $relation_type}]->(t)
                SET r.weight = $weight,
                    r.avg_strength = $avg_strength,
                    r.description = $rel_desc,
                    r.source_chunks = $source_chunks
            """,
                source_key=edge["source_key"],
                target_key=edge["target_key"],
                relation_type=edge["relation_type"],
                weight=edge["weight"],
                avg_strength=avg_strength,
                rel_desc=rel_desc,
                source_chunks=list(edge["source_chunks"]),
            )

    print(f"  Graph persisted to Neo4j ({len(node_map)} nodes, {len(edge_map)} edges)")


def get_entity_chunks(entity_key: str) -> list[str]:
    """Get chunk IDs associated with an entity."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity {key: $key}) RETURN e.chunk_ids AS chunk_ids",
            key=entity_key,
        )
        record = result.single()
        if record:
            return record["chunk_ids"] or []
    return []


def get_neighbors(entity_key: str, hop: int = 1) -> list[dict]:
    """Get neighboring entities within N hops."""
    driver = _get_driver()
    results = []
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity {key: $key})-[*1..{hop}]-(neighbor:Entity)
            RETURN DISTINCT neighbor.key AS key,
                   neighbor.name AS name,
                   neighbor.type AS type,
                   neighbor.summary AS summary
        """.format(hop=hop), key=entity_key)
        for record in result:
            results.append(dict(record))
    return results


def get_stats() -> dict:
    """Return graph statistics."""
    driver = _get_driver()
    with driver.session() as session:
        node_count = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        edge_count = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS n").single()["n"]
        type_counts = session.run("""
            MATCH (e:Entity)
            RETURN e.type AS type, count(e) AS n
            ORDER BY n DESC
        """)
        return {
            "nodes": node_count,
            "edges": edge_count,
            "types": [dict(r) for r in type_counts],
        }
