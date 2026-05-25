from __future__ import annotations

from typing import List, Dict

from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever


def _embed_query(query: str) -> List[float]:
	retriever = HybridRetriever(strict_hybrid=False)
	return retriever._embed(query)


def graph_local_context(
	query: str,
	top_k: int = 5,
	hop: int = 2,
) -> dict:
	embedding = _embed_query(query)
	if not embedding:
		return {"context_text": "", "sources": [], "chunk_ids": []}

	graph = GraphRetriever()
	data = graph.local_search(query_embedding=embedding, top_k=top_k, hop=hop)
	graph.close()

	chunk_ids = data.get("chunk_ids", []) or []
	sources = [{"chunk_id": str(cid)} for cid in chunk_ids if cid]

	return {
		"context_text": data.get("context_text", ""),
		"sources": sources,
		"chunk_ids": chunk_ids,
		"entities": data.get("entities", []),
		"relations": data.get("relations", []),
	}


def graph_global_context(
	query: str,
	top_k: int = 5,
) -> dict:
	embedding = _embed_query(query)
	if not embedding:
		return {"context_text": "", "sources": [], "chunk_ids": []}

	graph = GraphRetriever()
	data = graph.global_search(query_embedding=embedding, top_k=top_k)
	graph.close()

	chunk_ids = data.get("chunk_ids", []) or []
	sources = [{"chunk_id": str(cid)} for cid in chunk_ids if cid]

	return {
		"context_text": data.get("context_text", ""),
		"sources": sources,
		"chunk_ids": chunk_ids,
		"communities": data.get("communities", []),
	}
