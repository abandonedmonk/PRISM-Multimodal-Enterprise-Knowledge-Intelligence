""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Hybrid Retriever                                                            ║
    ║  Merges Qdrant vector search + Neo4j graph search results.                 ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Combines two retrieval paths:
    - Path 1: Qdrant vector search (semantic similarity)
    - Path 2: Neo4j graph search (entity relationships + community reports)
    Then merges, deduplicates, and reranks the results.

Usage:
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve(query, top_k=5)
"""

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


def _load_qdrant():
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(
            host=(
                (config.QDRANT_HOST if config else None)
                or os.getenv("QDRANT_HOST", "localhost")
            ),
            port=(
                (config.QDRANT_PORT if config else None)
                or int(os.getenv("QDRANT_PORT", "6333"))
            ),
        )
    except ImportError:
        return None


def _load_embedding_model():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding("BAAI/bge-small-en-v1.5")
    except ImportError:
        return None


class HybridRetriever:
    def __init__(
        self,
        qdrant_collection: str | None = None,
        top_k: int = 10,
        use_local_reranker: bool = True,
        strict_hybrid: bool = True,
    ):
        self.collection = (
            qdrant_collection
            or (config.QDRANT_COLLECTION if config else None)
            or os.getenv("QDRANT_COLLECTION", "prism_filings")
        )
        self.top_k = top_k
        self.strict_hybrid = strict_hybrid
        self._qdrant = None
        self._embed_model = None

        if use_local_reranker:
            try:
                from retrieval.reranker import LocalReranker
                self._reranker = LocalReranker()
            except ImportError:
                self._reranker = None
        else:
            try:
                from retrieval.reranker import Reranker
                self._reranker = Reranker()
            except ImportError:
                self._reranker = None

        try:
            from retrieval.graph_retriever import GraphRetriever
            self._graph_retriever = GraphRetriever()
        except Exception:
            self._graph_retriever = None

    def _get_qdrant(self):
        if self._qdrant is None:
            self._qdrant = _load_qdrant()
        return self._qdrant

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = _load_embedding_model()
        return self._embed_model

    def _embed(self, text: str) -> list[float]:
        model = self._get_embed_model()
        if model is None:
            return []
        try:
            embeds = list(model.embed([text]))
            return embeds[0].tolist() if embeds else []
        except Exception:
            return []

    def _build_qdrant_filter(self, metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            return None

        conditions = []
        field_map = {
            "ticker": "ticker",
            "filing_type": "filing_type",
            "year": "year",
            "quarter": "quarter",
            "section": "section",
        }
        for key, qdrant_field in field_map.items():
            val = metadata_filter.get(key)
            if val is not None and val != "":
                conditions.append(
                    FieldCondition(
                        key=qdrant_field,
                        match=MatchValue(value=str(val) if key in ("year", "quarter") else val),
                    )
                )

        if not conditions:
            return None
        if len(conditions) == 1:
            return Filter(must=[conditions[0]])
        return Filter(must=conditions)

    def _qdrant_search(
        self,
        query_embedding: list[float],
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        client = self._get_qdrant()
        if client is None:
            return []

        qdrant_filter = self._build_qdrant_filter(metadata_filter)

        try:
            response = client.query_points(
                collection_name=self.collection,
                query=query_embedding,
                using="dense",
                limit=limit,
                query_filter=qdrant_filter,
            )
            points = response.points if hasattr(response, "points") else response
            return [
                {
                    **r.payload,
                    "score": r.score,
                    "id": str(r.id),
                }
                for r in points
            ]
        except Exception as e:
            print(f"  Qdrant search error: {e}")
            return []

    def _qdrant_sparse_search(self, query_text: str, limit: int) -> list[dict]:
        client = self._get_qdrant()
        if client is None:
            return []

        try:
            response = client.query_points(
                collection_name=self.collection,
                query=[],
                using="sparse",
                limit=limit,
                query_filter=None,
            )
            points = response.points if hasattr(response, "points") else response
            return [
                {
                    **r.payload,
                    "score": r.score,
                    "id": str(r.id),
                }
                for r in points
            ]
        except Exception:
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = True,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Perform hybrid retrieval combining Qdrant vector search + Neo4j graph.

        Args:
            query: User query string
            top_k: Number of final results to return
            hybrid: If True, include graph search results; if False, vector only
            metadata_filter: Optional filter dict with ticker, filing_type, year, quarter, section

        Returns:
            List of chunk dicts, deduplicated, sorted by relevance score
        """
        if hybrid and self.strict_hybrid and not self._graph_retriever:
            raise RuntimeError(
                "Graph retriever unavailable. Start Neo4j and ensure graph data is loaded."
            )

        query_embedding = self._embed(query)
        if not query_embedding:
            raise RuntimeError(
                "Query embedding failed. Ensure embedding model/service is available."
            )

        vector_results = self._qdrant_search(query_embedding, limit=self.top_k * 2, metadata_filter=metadata_filter)
        if self.strict_hybrid and not vector_results:
            raise RuntimeError(
                "Vector retrieval returned no results. Ensure Qdrant is running and indexed."
            )

        seen_chunk_ids = {}
        for r in vector_results:
            cid = r.get("chunk_id", r.get("id", ""))
            if cid:
                seen_chunk_ids[cid] = r

        if hybrid and self._graph_retriever:
            try:
                local_ctx = self._graph_retriever.local_search(
                    query_embedding=query_embedding,
                    top_k=10,
                    hop=2,
                )

                for chunk_id in local_ctx.get("chunk_ids", []):
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids[chunk_id] = {
                            "chunk_id": chunk_id,
                            "text": f"[From graph context] {local_ctx.get('context_text', '')[:500]}",
                            "source": "graph",
                            "score": 0.5,
                        }

            except Exception as e:
                if self.strict_hybrid:
                    raise RuntimeError(f"Graph retrieval failed: {e}") from e
                print(f"  Graph search warning: {e}")

        all_results = list(seen_chunk_ids.values())

        if self._reranker:
            try:
                all_results = self._reranker.rerank(query, all_results, top_k=top_k * 2)
            except Exception as e:
                print(f"  Reranking warning: {e}")
                all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)

        final = all_results[:top_k]
        if self.strict_hybrid and not final:
            raise RuntimeError("Hybrid retrieval produced no final results.")
        return final

    def retrieve_global(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """Global search: use community reports for thematic queries.

        Args:
            query: User query string
            top_k: Number of community reports to retrieve

        Returns:
            Dict with community context text and associated chunk IDs
        """
        if not self._graph_retriever:
            if self.strict_hybrid:
                raise RuntimeError("Graph retriever unavailable for global retrieval.")
            return {"context_text": "", "chunk_ids": []}

        query_embedding = self._embed(query)
        if not query_embedding:
            raise RuntimeError(
                "Query embedding failed. Ensure embedding model/service is available."
            )

        result = self._graph_retriever.global_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        if self.strict_hybrid and not result.get("chunk_ids"):
            raise RuntimeError(
                "Global graph retrieval returned no chunk IDs. Generate community reports and entity/community embeddings."
            )
        return result

    def combined_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> dict:
        """Run all retrieval paths and merge results.

        Returns a dict with:
            - qdrant_results: list of vector search chunks
            - graph_local: dict from local_search
            - graph_global: dict from global_search
            - merged_chunks: deduplicated list of all chunks
            - context_text: assembled text from all sources
            - sources: list of SourceItem dicts
            - stats: counts of items from each source
        """
        query_embedding = self._embed(query)
        if not query_embedding:
            return {
                "qdrant_results": [],
                "graph_local": {},
                "graph_global": {},
                "merged_chunks": [],
                "context_text": "",
                "sources": [],
                "stats": {"qdrant_chunks": 0, "graph_entities": 0, "communities": 0},
            }

        qdrant_results = self._qdrant_search(query_embedding, limit=top_k * 2, metadata_filter=metadata_filter)

        graph_local = {}
        graph_global = {}
        if self._graph_retriever:
            try:
                graph_local = self._graph_retriever.local_search(query_embedding=query_embedding, top_k=top_k, hop=2)
            except Exception as e:
                print(f"  Graph local search warning: {e}")
            try:
                graph_global = self._graph_retriever.global_search(query_embedding=query_embedding, top_k=top_k)
            except Exception as e:
                print(f"  Graph global search warning: {e}")

        seen_ids = {}
        for r in qdrant_results:
            cid = r.get("chunk_id", r.get("id", ""))
            if cid:
                r["_source"] = "qdrant"
                seen_ids[cid] = r

        for cid in graph_local.get("chunk_ids", []):
            if cid and cid not in seen_ids:
                seen_ids[cid] = {
                    "chunk_id": cid,
                    "text": graph_local.get("context_text", "")[:1000],
                    "source": "graph_local",
                    "score": 0.5,
                    "_source": "graph_local",
                }

        for cid in graph_global.get("chunk_ids", []):
            if cid and cid not in seen_ids:
                seen_ids[cid] = {
                    "chunk_id": cid,
                    "text": graph_global.get("context_text", "")[:1000],
                    "source": "graph_global",
                    "score": 0.4,
                    "_source": "graph_global",
                }

        merged_chunks = list(seen_ids.values())

        if self._reranker and merged_chunks:
            try:
                merged_chunks = self._reranker.rerank(query, merged_chunks, top_k=len(merged_chunks))
            except Exception:
                merged_chunks = sorted(merged_chunks, key=lambda x: x.get("score", 0), reverse=True)

        MAX_CHUNKS = top_k * 2
        MAX_CHUNK_CHARS = 800
        MAX_GRAPH_CHARS = 2000
        MAX_TOTAL_CHARS = 30000

        context_parts = []
        total_chars = 0

        for r in merged_chunks[:MAX_CHUNKS]:
            text = r.get("text", "")
            if text:
                text = text[:MAX_CHUNK_CHARS]
                if total_chars + len(text) > MAX_TOTAL_CHARS:
                    break
                context_parts.append(text)
                total_chars += len(text)

        if graph_local.get("context_text") and total_chars < MAX_TOTAL_CHARS:
            graph_text = graph_local["context_text"][:MAX_GRAPH_CHARS]
            if total_chars + len(graph_text) <= MAX_TOTAL_CHARS:
                context_parts.append(f"[Graph Local Context]\n{graph_text}")
                total_chars += len(graph_text)

        if graph_global.get("context_text") and total_chars < MAX_TOTAL_CHARS:
            graph_text = graph_global["context_text"][:MAX_GRAPH_CHARS]
            if total_chars + len(graph_text) <= MAX_TOTAL_CHARS:
                context_parts.append(f"[Graph Global Context]\n{graph_text}")
                total_chars += len(graph_text)

        sources = []
        for r in merged_chunks[:MAX_CHUNKS]:
            src = {}
            for key in ("ticker", "filing_type", "year", "quarter", "section", "source"):
                val = r.get(key)
                if val:
                    src[key] = str(val) if key in ("year",) else val
            cid = r.get("chunk_id") or r.get("id")
            if cid:
                src["chunk_id"] = str(cid)
            score = r.get("score") or r.get("rerank_score")
            if score is not None:
                src["score"] = float(score)
            _src = r.get("_source", "unknown")
            if _src:
                src["retrieval_source"] = _src
            if src:
                sources.append(src)

        return {
            "qdrant_results": qdrant_results[:MAX_CHUNKS],
            "graph_local": graph_local,
            "graph_global": graph_global,
            "merged_chunks": merged_chunks[:MAX_CHUNKS],
            "context_text": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "stats": {
                "qdrant_chunks": len(qdrant_results[:MAX_CHUNKS]),
                "graph_entities": len(graph_local.get("entities", [])),
                "communities": len(graph_global.get("communities", [])),
            },
        }
