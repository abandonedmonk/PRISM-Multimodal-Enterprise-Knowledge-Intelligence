import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException

from api.models import ChatRequest, ChatResponse, SourceItem, ContextUsed


router = APIRouter(prefix="/api", tags=["chat"])


VALID_MODES = {"combined", "hybrid", "graph_local", "graph_global", "vector", "auto"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
	start = time.time()

	if request.retrieval_mode not in VALID_MODES:
		raise HTTPException(
			status_code=400,
			detail=f"Invalid retrieval_mode. Must be one of: {', '.join(VALID_MODES)}"
		)

	metadata_filter = None
	if request.metadata_filter:
		mf = request.metadata_filter
		metadata_filter = mf.model_dump(exclude_none=True)

	try:
		from agents.orchestrator import run_agent

		result = run_agent(
			query=request.query,
			chat_history=request.chat_history if request.chat_history else None,
			top_k=request.top_k,
			use_web=request.use_web,
			use_vision=request.use_vision,
			retrieval_mode=request.retrieval_mode,
			metadata_filter=metadata_filter,
			temperature=request.temperature,
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}") from e

	if "answer" not in result:
		raise HTTPException(status_code=500, detail="Agent returned no answer")

	sources_raw = result.get("sources", [])
	sources = []
	for s in sources_raw:
		if isinstance(s, dict):
			clean = {k: v for k, v in s.items() if v is not None}
			sources.append(SourceItem(**clean))

	context_stats = result.get("context_stats", {})
	total_sources = len(sources)

	latency_ms = (time.time() - start) * 1000

	return ChatResponse(
		answer=result.get("answer", ""),
		sources=sources,
		tool_trace=result.get("tool_trace", []),
		context_used=ContextUsed(
			qdrant_chunks=context_stats.get("qdrant_chunks", 0),
			graph_entities=context_stats.get("graph_entities", 0),
			communities=context_stats.get("communities", 0),
			total_sources=total_sources,
		),
		retrieval_mode=result.get("route", request.retrieval_mode),
		latency_ms=round(latency_ms, 2),
	)


@router.get("/chat/history/clear")
def clear_history():
	return {"status": "ok", "message": "History cleared (stateless - no persistence)"}