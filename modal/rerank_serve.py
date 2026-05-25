"""Cross-encoder Reranker — FastAPI on Modal (CPU).

Serves BGE reranker on CPU via OpenAI-compatible `/v1/rerank` endpoint.
No GPU needed — MiniLM / BGE reranker is tiny and fast on CPU.

Usage:
    modal deploy modal/rerank_serve.py        # deploy
    modal run modal/rerank_serve.py           # test locally
"""

import modal

MINUTES = 60

RERANK_MODEL = "BAAI/bge-reranker-base-v2.0"

rerank_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]>=0.115.0", "fastembed>=0.4.0")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("prism-rerank-serve")

RERANK_PORT = 8000


@app.function(
    image=rerank_image,
    cpu=1.0,
    memory=512,
    scaledown_window=15 * MINUTES,
    timeout=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=RERANK_PORT, startup_timeout=5 * MINUTES)
def serve():
    import time

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastembed import Rerank

    _t0 = time.time()
    model = Rerank(model_name=RERANK_MODEL)
    _load_time = time.time() - _t0
    print(f"Loaded {RERANK_MODEL} in {_load_time:.1f}s")

    api = FastAPI(title="PRISM Reranker Service")

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": RERANK_MODEL}

    @api.post("/v1/rerank")
    async def rerank(request: Request):
        body = await request.json()
        query = body.get("query", "")
        passages = body.get("passages", [])
        top_k = body.get("top_k", min(10, len(passages)))
        model_name = body.get("model", RERANK_MODEL)

        if not query or not passages:
            return JSONResponse(
                {"results": [], "model": model_name, "error": "query and passages required"}
            )

        results = model.rerank(query=query, passages=passages, top_k=top_k)

        rerank_results = []
        for i, score in enumerate(results):
            rerank_results.append({
                "index": i,
                "relevance_score": float(score),
                "text": passages[i],
            })

        return JSONResponse({
            "results": rerank_results,
            "model": model_name,
        })


@app.local_entrypoint()
async def main():
    import aiohttp

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        print("Health check...")
        async with session.get("/health", timeout=5 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed!")

        payload = {
            "query": "What were NVIDIA's revenue figures?",
            "passages": [
                "NVIDIA reported $60.9B revenue in FY2024, up 122% year-over-year.",
                "Apple's iPhone sales reached $200B in 2023.",
                "NVIDIA's data center segment grew to $47.5B, driven by AI demand.",
                "The company paid $500M in dividends last quarter.",
                "NVIDIA competes with AMD in the GPU market.",
            ],
            "top_k": 3,
            "model": RERANK_MODEL,
        }
        async with session.post("/v1/rerank", json=payload) as resp:
            result = await resp.json()
            print(f"Reranked results:")
            for item in result["results"]:
                print(f"  [{item['relevance_score']:.3f}] {item['text'][:80]}...")
            print(f"  Model: {result['model']}")