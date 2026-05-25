import time
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from api.models import (
	RetrievalRequest,
	RetrievalResponse,
	SourceItem,
	MetadataFilter,
)


router = APIRouter(prefix="/api", tags=["retrieval"])


def _get_retriever():
	from retrieval.hybrid_retriever import HybridRetriever
	return HybridRetriever(strict_hybrid=False)


def _get_graph_retriever():
	from retrieval.graph_retriever import GraphRetriever
	return GraphRetriever()


@router.get("/retrieve", response_model=RetrievalResponse)
def direct_retrieve(
	q: str = Query(..., min_length=1, description="Query string"),
	mode: str = Query(default="combined", description="Mode: combined, hybrid, vector, graph_local, graph_global"),
	top_k: int = Query(default=5, ge=1, le=20),
	ticker: str | None = Query(None, description="Filter by ticker"),
	filing_type: str | None = Query(None, description="Filter by filing type"),
	year: int | None = Query(None, description="Filter by year"),
	quarter: str | None = Query(None, description="Filter by quarter"),
	section: str | None = Query(None, description="Filter by section"),
):
	start = time.time()

	metadata_filter = None
	if any([ticker, filing_type, year, quarter, section]):
		metadata_filter = MetadataFilter(
			ticker=ticker,
			filing_type=filing_type,
			year=year,
			quarter=quarter,
			section=section,
		)

	retriever = _get_retriever()
	graph = _get_graph_retriever()

	if mode == "graph_local":
		embedding = retriever._embed(q)
		if not embedding:
			raise HTTPException(status_code=500, detail="Embedding failed")
		data = graph.local_search(query_embedding=embedding, top_k=top_k, hop=2)
		graph.close()
		chunks = []
		for cid in data.get("chunk_ids", []):
			chunks.append({"chunk_id": cid, "text": data.get("context_text", ""), "source": "graph_local"})
		sources = [{"chunk_id": str(cid)} for cid in data.get("chunk_ids", []) if cid]
		graph_ctx = None
	elif mode == "graph_global":
		embedding = retriever._embed(q)
		if not embedding:
			raise HTTPException(status_code=500, detail="Embedding failed")
		data = graph.global_search(query_embedding=embedding, top_k=top_k)
		graph.close()
		chunks = []
		for cid in data.get("chunk_ids", []):
			chunks.append({"chunk_id": cid, "text": data.get("context_text", ""), "source": "graph_global"})
		sources = [{"chunk_id": str(cid)} for cid in data.get("chunk_ids", []) if cid]
		graph_ctx = data
	elif mode == "vector":
		mf_dict = metadata_filter.model_dump(exclude_none=True) if metadata_filter else None
		results = retriever.retrieve(q, top_k=top_k, hybrid=False, metadata_filter=mf_dict)
		chunks = results
		sources = [{"chunk_id": r.get("chunk_id") or r.get("id"), "score": r.get("score")} for r in results if r.get("chunk_id") or r.get("id")]
		graph_ctx = None
	elif mode == "hybrid":
		mf_dict = metadata_filter.model_dump(exclude_none=True) if metadata_filter else None
		results = retriever.retrieve(q, top_k=top_k, hybrid=True, metadata_filter=mf_dict)
		chunks = results
		sources = [{"chunk_id": r.get("chunk_id") or r.get("id"), "score": r.get("score")} for r in results if r.get("chunk_id") or r.get("id")]
		graph_ctx = None
	else:
		mf_dict = metadata_filter.model_dump(exclude_none=True) if metadata_filter else None
		data = retriever.combined_search(q, top_k=top_k, metadata_filter=mf_dict)
		chunks = data.get("merged_chunks", [])
		sources = data.get("sources", [])
		graph_ctx = {"local": data.get("graph_local", {}), "global": data.get("graph_global", {})}

	latency_ms = (time.time() - start) * 1000
	return RetrievalResponse(
		chunks=chunks,
		graph_context=graph_ctx,
		sources=[SourceItem(**s) if isinstance(s, dict) else s for s in sources],
		retrieval_mode=mode,
		latency_ms=round(latency_ms, 2),
	)


@router.get("/graph/search")
def graph_search(
	q: str = Query(..., description="Query for graph search"),
	mode: str = Query(default="local", description="local or global"),
	top_k: int = Query(default=5, ge=1, le=20),
):
	start = time.time()

	retriever = _get_retriever()
	graph = _get_graph_retriever()
	embedding = retriever._embed(q)

	if not embedding:
		raise HTTPException(status_code=500, detail="Embedding failed")

	if mode == "global":
		data = graph.global_search(query_embedding=embedding, top_k=top_k)
	else:
		data = graph.local_search(query_embedding=embedding, top_k=top_k, hop=2)

	graph.close()

	latency_ms = (time.time() - start) * 1000
	return {
		"query": q,
		"mode": mode,
		"data": data,
		"latency_ms": round(latency_ms, 2),
	}


@router.get("/health")
def health_check():
	results = {"qdrant": "unknown", "neo4j": "unknown", "embedding": "unknown"}

	try:
		from retrieval.hybrid_retriever import _load_qdrant
		client = _load_qdrant()
		if client:
			client.get_collection("prism_filings")
			results["qdrant"] = "ok"
	except Exception as e:
		results["qdrant"] = f"error: {str(e)[:100]}"

	try:
		from retrieval.graph_retriever import _get_driver
		driver = _get_driver()
		with driver.session() as session:
			session.run("MATCH (n) RETURN count(n) as cnt LIMIT 1")
		driver.close()
		results["neo4j"] = "ok"
	except Exception as e:
		results["neo4j"] = f"error: {str(e)[:100]}"

	try:
		retriever = _get_retriever()
		emb = retriever._embed("test")
		if emb:
			results["embedding"] = "ok"
	except Exception as e:
		results["embedding"] = f"error: {str(e)[:100]}"

	return results