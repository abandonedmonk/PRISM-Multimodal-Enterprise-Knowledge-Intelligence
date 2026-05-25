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

import networkx as nx
from networkx.algorithms.community import louvain_communities


def _get_driver():
    try:
        neo4j = importlib.import_module("neo4j")
    except Exception as exc:
        raise ImportError("neo4j is required for community detection") from exc

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


def detect_communities(
    min_community_size: int = 2,
    relationship_type: str = "RELATES",
) -> list[dict]:
    driver = _get_driver()

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS n")
        node_count = result.single()["n"]
        if node_count == 0:
            print("  No entities in graph. Skipping community detection.")
            return []

    print(f"  Loading graph from Neo4j for Python Louvain...")

    G = nx.Graph()

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.key AS key")
        for record in result:
            G.add_node(record["key"])

    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE type(r) = $relationship_type
            RETURN s.key AS source, t.key AS target, r.weight AS weight
        """, relationship_type=relationship_type)
        for record in result:
            weight = record["weight"]
            if weight is None:
                weight = 1
            G.add_edge(record["source"], record["target"], weight=int(weight))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() == 0:
        print("  No edges in graph. Cannot detect communities.")
        return []

    print("  Running Louvain community detection (NetworkX)...")
    detected_communities = louvain_communities(G, weight="weight", resolution=1.0)
    partition = {
        node: comm_id
        for comm_id, nodes in enumerate(detected_communities)
        for node in nodes
    }

    community_map: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        if comm_id not in community_map:
            community_map[comm_id] = []
        community_map[comm_id].append(node)

    communities = []
    community_counter = 0
    for comm_id in sorted(community_map.keys()):
        node_keys = community_map[comm_id]
        size = len(node_keys)
        if size < min_community_size:
            continue

        cid = f"community_{community_counter}"
        communities.append({
            "community_id": cid,
            "level": 0,
            "size": size,
            "node_keys": node_keys,
            "raw_community_id": comm_id,
        })
        community_counter += 1

    print(f"  Detected {len(communities)} communities")

    with driver.session() as session:
        for comm in communities:
            session.run("""
                MERGE (c:Community {community_id: $community_id})
                SET c.level = $level,
                    c.member_count = $size
            """,
                community_id=comm["community_id"],
                level=comm["level"],
                size=comm["size"],
            )

            for node_key in comm["node_keys"]:
                session.run("""
                    MATCH (e:Entity {key: $key})
                    MATCH (c:Community {community_id: $cid})
                    MERGE (c)-[:HAS_MEMBER]->(e)
                    SET e.community_id = $cid
                """, key=node_key, cid=comm["community_id"])

    print(f"  Communities persisted to Neo4j")
    return communities


def get_community_entities(community_id: str, top_n: int = 20) -> list[dict]:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Community {community_id: $cid})-[:HAS_MEMBER]->(e:Entity)
            RETURN e.key AS key, e.name AS name, e.type AS type,
                   e.summary AS summary, e.occurrence_count AS count
            ORDER BY count DESC
            LIMIT $top_n
        """, cid=community_id, top_n=top_n)
        return [dict(r) for r in result]


def get_community_relations(community_id: str, top_n: int = 30) -> list[dict]:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Community {community_id: $cid})-[:HAS_MEMBER]->(e1:Entity)
            MATCH (e1)-[r:RELATES]->(e2:Entity)
            MATCH (c)-[:HAS_MEMBER]->(e2)
            RETURN e1.name AS source, e2.name AS target,
                   r.type AS relation_type, r.description AS description,
                   r.weight AS weight
            ORDER BY r.weight DESC
            LIMIT $top_n
        """, cid=community_id, top_n=top_n)
        return [dict(r) for r in result]
