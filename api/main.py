import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, retrieval


app = FastAPI(
	title="PRISM API",
	description="Hybrid RAG API for SEC filings - combines Qdrant vector search + Neo4j GraphRAG",
	version="1.0.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(retrieval.router)


@app.get("/", tags=["root"])
def root():
	return {
		"service": "PRISM API",
		"version": "1.0.0",
		"endpoints": {
			"chat": "POST /api/chat",
			"retrieve": "GET /api/retrieve",
			"graph_search": "GET /api/graph/search",
			"health": "GET /api/health",
		},
	}


@app.get("/health", tags=["health"])
def health():
	qdrant_status = "unknown"
	neo4j_status = "unknown"
	embedding_status = "unknown"

	try:
		from retrieval.hybrid_retriever import _load_qdrant
		client = _load_qdrant()
		if client:
			client.get_collection("prism_filings")
			qdrant_status = "ok"
	except Exception as e:
		qdrant_status = f"error: {str(e)[:80]}"

	try:
		from retrieval.graph_retriever import _get_driver
		driver = _get_driver()
		with driver.session() as session:
			result = session.run("MATCH (n) RETURN count(n) as cnt LIMIT 1")
			result.single()
		driver.close()
		neo4j_status = "ok"
	except Exception as e:
		neo4j_status = f"error: {str(e)[:80]}"

	try:
		from retrieval.hybrid_retriever import HybridRetriever
		retriever = HybridRetriever(strict_hybrid=False)
		emb = retriever._embed("test")
		embedding_status = "ok" if emb else "failed"
	except Exception as e:
		embedding_status = f"error: {str(e)[:80]}"

	overall = "ok" if all(s == "ok" for s in [qdrant_status, neo4j_status, embedding_status]) else "degraded"

	return {
		"status": overall,
		"services": {
			"qdrant": qdrant_status,
			"neo4j": neo4j_status,
			"embedding": embedding_status,
		},
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)