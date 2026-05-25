"""BGE-small + Splade Embedding — FastAPI on Modal (CPU).

OpenAI-compatible /v1/embeddings (dense) and /v1/sparse_embeddings (sparse).
Runs on CPU via fastembed — no GPU/CUDA needed.

Usage:
    modal deploy modal/embed_serve.py        # deploy
    modal run modal/embed_serve.py           # test locally
"""

import modal

MINUTES = 60

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

SERVER_CODE = '''
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding, SparseTextEmbedding

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

_t0 = time.time()
dense_model = TextEmbedding(model_name=DENSE_MODEL)
print(f"Loaded {DENSE_MODEL} in {time.time() - _t0:.1f}s")

_t1 = time.time()
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
print(f"Loaded {SPARSE_MODEL} in {time.time() - _t1:.1f}s")

api = FastAPI(title="PRISM Embedding Service")

@api.get("/health")
async def health():
    return {"status": "ok", "models": [DENSE_MODEL, SPARSE_MODEL]}

@api.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    embeds = list(dense_model.embed(inputs))
    data = [{"object": "embedding", "index": i, "embedding": emb.tolist()} for i, emb in enumerate(embeds)]
    usage = {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": sum(len(t.split()) for t in inputs)}
    return JSONResponse({"object": "list", "data": data, "model": DENSE_MODEL, "usage": usage})

@api.post("/v1/sparse_embeddings")
async def sparse_embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    embeds = list(sparse_model.embed(inputs))
    data = [{"object": "sparse_embedding", "index": i, "indices": emb.indices.tolist(), "values": emb.values.tolist()} for i, emb in enumerate(embeds)]
    usage = {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": sum(len(t.split()) for t in inputs)}
    return JSONResponse({"object": "list", "data": data, "model": SPARSE_MODEL, "usage": usage})
'''

embed_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]>=0.115.0", "fastembed>=0.4.0")
    .run_commands(f"cat > /root/server.py << 'PYEOF'\n{SERVER_CODE}\nPYEOF")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("prism-embed-serve")

EMBED_PORT = 8000


@app.function(
    image=embed_image,
    cpu=1.0,
    memory=2048,
    scaledown_window=15 * MINUTES,
    timeout=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=EMBED_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = f"uvicorn server:api --host 0.0.0.0 --port {EMBED_PORT}"
    print(f"Starting: {cmd}")
    subprocess.Popen(cmd, shell=True)


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

        payload = {"input": ["Apple Inc. reported $394B revenue"], "model": DENSE_MODEL}
        async with session.post("/v1/embeddings", json=payload) as resp:
            result = await resp.json()
            for item in result["data"]:
                print(f"  Dense dim={len(item['embedding'])}")

        async with session.post("/v1/sparse_embeddings", json=payload) as resp:
            result = await resp.json()
            for item in result["data"]:
                print(f"  Sparse non-zero={len(item['indices'])}")
