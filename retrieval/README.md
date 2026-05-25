# PRISM Retrieval

Query-time retrieval for PRISM. This package reads indexes built by `ingestion/` and returns context for agents/API routes.

## Boundary

- `ingestion/` is the offline/write path: it creates chunks, writes Qdrant vectors, builds the Neo4j graph, generates community reports, and stores graph embeddings.
- `retrieval/` is the online/read path: it searches Qdrant, queries Neo4j, merges context, and reranks results.
- Neo4j query code belongs here only when it answers user queries. Neo4j mutation/index-building code belongs in `ingestion/`.

## Quick Start

```bash
# Ensure Qdrant + Neo4j are populated (see ingestion/ README)
docker-compose up -d
```

```python
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.graph_retriever import GraphRetriever

# Hybrid (Qdrant + Neo4j + reranking)
retriever = HybridRetriever()
results = retriever.retrieve("What are NVIDIA's major data center risks?", top_k=5)
global_context = retriever.retrieve_global("What themes appear across the filing?", top_k=5)

# Graph-only
graph = GraphRetriever()
local = graph.local_search("NVIDIA data center revenue", top_k=5)
global_ctx = graph.global_search("Cross-company risk themes", top_k=5)
```

## Modules

| Module | Purpose |
|--------|---------|
| `hybrid_retriever.py` | Combines Qdrant semantic search with Neo4j graph context and reranking |
| `graph_retriever.py` | Local entity-neighborhood search and global community-report search over Neo4j |
| `reranker.py` | Cross-encoder reranking (local CrossEncoder or HTTP reranker service) |

## Required Indexes

- **Qdrant** collection populated by `python -m ingestion.scripts.ingest`
- **Neo4j** graph populated by `python -m ingestion.scripts.run_neo4j_pipeline`

If Neo4j is unavailable, `HybridRetriever` falls back to vector-only retrieval.
