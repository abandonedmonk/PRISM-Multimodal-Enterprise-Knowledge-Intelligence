"""Qwen 2.5 14B Instruct — vLLM on Modal (A100 40GB, FP16).

OpenAI-compatible endpoint for entity extraction, community reports,
and table summarization. Deploys as a serverless web server.

Usage:
    modal deploy modal/llm_serve.py        # deploy
    modal run modal/llm_serve.py           # test locally
"""

import modal

MINUTES = 60
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("prism-llm-serve")


def _download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_NAME,
        ignore_patterns=["*.pt", "*.gguf"],
    )


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub[hf_xet]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "5",
        }
    )
    .run_function(
        _download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=10 * MINUTES,
    )
)

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="A100",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--no-enforce-eager",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def main():
    import aiohttp

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        print("Health check...")
        async with session.get("/health", timeout=10 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed!")

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Extract entities from: Apple Inc. reported $394B revenue."},
            ],
            "model": MODEL_NAME,
            "max_tokens": 200,
        }
        async with session.post(
            "/v1/chat/completions", json=payload
        ) as resp:
            result = await resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
