""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Reranker                                                                   ║
    ║  CrossEncoder-based reranking of retrieved chunks.                         ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Rerank retrieved chunks from hybrid search using a CrossEncoder model.
    Scores each chunk against the query to produce a relevance-ordered list.

Usage:
    from retrieval.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query, chunks, top_k=5)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config as _cfg
except ImportError:
    _cfg = None

import os


DEFAULT_RERANKER_URL = "http://localhost:8002"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
    ):
        self.endpoint = (
            endpoint
            or (_cfg.RERANKER_SERVICE_URL if _cfg else None)
            or os.getenv("RERANKER_SERVICE_URL", DEFAULT_RERANKER_URL)
        )
        self.model = (
            model
            or os.getenv("RERANKER_MODEL", DEFAULT_MODEL)
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=30.0)
            except ImportError:
                return None
        return self._client

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank chunks by relevance to the query using CrossEncoder.

        Args:
            query: User query string
            chunks: List of chunk dicts from hybrid retriever
            top_k: Number of top results to return

        Returns:
            List of reranked chunk dicts with added "rerank_score" field,
            sorted by descending score

        Example:
            reranked = reranker.rerank(
                "What was NVIDIA's revenue growth?",
                chunks,  # from hybrid retriever
                top_k=5
            )
            # Returns top 5 chunks sorted by CrossEncoder score
        """
        if not chunks:
            return []

        client = self._get_client()
        if client is None:
            return chunks[:top_k]

        try:
            import httpx

            texts = [
                {
                    "query": query,
                    "text": c.get("text", "")[:512],
                }
                for c in chunks
            ]

            resp = client.post(
                f"{self.endpoint}/rerank",
                json={"query": query, "texts": texts},
            )

            if resp.status_code == 200:
                scores = resp.json().get("scores", [])
                for i, c in enumerate(chunks):
                    c["rerank_score"] = scores[i] if i < len(scores) else 0.0
            else:
                for c in chunks:
                    c["rerank_score"] = 0.0

        except Exception:
            for c in chunks:
                c["rerank_score"] = 0.0

        return sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)[:top_k]


class LocalReranker:
    """Fastembed Rerank-based reranker that runs locally (no separate service needed)."""

    def __init__(self, model: str | None = None):
        model = model or "BAAI/bge-reranker-base-v2.0"
        self._model = None
        self._available = False
        try:
            from fastembed import Rerank
            self._model = Rerank(model_name=model)
            self._available = True
        except ImportError:
            self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not chunks or not self._available:
            return chunks[:top_k]

        try:
            passages = [c.get("text", "")[:512] for c in chunks]
            scores = list(self._model.rerank(query=query, passages=passages, top_k=len(passages)))

            for i, c in enumerate(chunks):
                c["rerank_score"] = float(scores[i])
        except Exception:
            for c in chunks:
                c["rerank_score"] = 0.0

        return sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)[:top_k]