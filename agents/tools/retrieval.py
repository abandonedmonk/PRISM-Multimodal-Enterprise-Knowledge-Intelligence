from __future__ import annotations

from typing import List, Dict, Optional

from retrieval.hybrid_retriever import HybridRetriever


def _build_sources(results: List[dict]) -> List[Dict[str, str]]:
	sources: List[Dict[str, str]] = []
	for r in results:
		item: Dict[str, str] = {}
		for key in ("ticker", "filing_type", "year", "quarter", "source"):
			val = r.get(key)
			if val:
				item[key] = str(val)
		chunk_id = r.get("chunk_id") or r.get("id")
		if chunk_id:
			item["chunk_id"] = str(chunk_id)
		if item:
			sources.append(item)
	return sources


def _metadata_filter_to_dict(metadata_filter) -> dict | None:
	if metadata_filter is None:
		return None
	if hasattr(metadata_filter, "model_dump"):
		return metadata_filter.model_dump(exclude_none=True)
	if isinstance(metadata_filter, dict):
		return {k: v for k, v in metadata_filter.items() if v is not None}
	return None


def retrieve_chunks(
	query: str,
	top_k: int = 5,
	hybrid: bool = True,
	metadata_filter=None,
) -> dict:
	retriever = HybridRetriever(strict_hybrid=False)
	filter_dict = _metadata_filter_to_dict(metadata_filter)
	results = retriever.retrieve(query, top_k=top_k, hybrid=hybrid, metadata_filter=filter_dict)

	context_parts = []
	for r in results:
		text = r.get("text", "")
		if text:
			context_parts.append(text.strip())

	return {
		"context_text": "\n\n".join(context_parts[:top_k]),
		"sources": _build_sources(results),
		"chunks": results,
	}


def retrieve_combined(
	query: str,
	top_k: int = 5,
	metadata_filter=None,
) -> dict:
	retriever = HybridRetriever(strict_hybrid=False)
	filter_dict = _metadata_filter_to_dict(metadata_filter)
	data = retriever.combined_search(query, top_k=top_k, metadata_filter=filter_dict)
	return {
		"context_text": data.get("context_text", ""),
		"sources": data.get("sources", []),
		"chunks": data.get("merged_chunks", []),
		"qdrant_results": data.get("qdrant_results", []),
		"graph_local": data.get("graph_local", {}),
		"graph_global": data.get("graph_global", {}),
		"stats": data.get("stats", {}),
	}
