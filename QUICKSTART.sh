#!/bin/bash
# PRISM Complete Setup & Query Guide
# Copy-paste each section in order

echo "=== PHASE 1: Setup (Run Once) ==="
cd /path/to/PRISM

# Create venv
python3.10 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install packages
pip install -r requirements.txt
playwright install chromium

# Copy config
cp .env.example .env
echo "👉 Edit .env — set NEO4J_PASSWORD at minimum"

echo ""
echo "=== PHASE 2: Start Services (Leave Running) ==="
# Terminal 1 — start Docker
docker-compose up -d
docker-compose ps  # wait for both to show "healthy"

echo ""
echo "=== PHASE 3: Deploy LLM (Choose One) ==="

echo "Option A: Modal (Recommended)"
modal login
modal deploy modal/llm_serve.py &
modal deploy modal/vision_serve.py &
modal deploy modal/embed_serve.py &
modal deploy modal/rerank_serve.py &
wait
echo "👉 Copy returned URLs into config.yaml"

echo ""
echo "Option B: Local vLLM (needs GPU)"
# Terminal 2
# python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen2.5-14B-Instruct \
#   --port 8000 \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.90

# Terminal 3
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-3.2-11B-Vision-Instruct \
#   --port 8001 \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.92 \
#   --limit-mm-per-prompt image=5

echo ""
echo "=== PHASE 4: Download & Process Data ==="

# Download filings (one-time)
python -m ingestion.scripts.download_filings --tickers NVDA AAPL

# Process into chunks
python -m ingestion.scripts.process_all_filings --force

# Index in Qdrant
python -m ingestion.scripts.ingest --recreate

# Build Neo4j graph (takes 5-15 min)
python -m ingestion.scripts.run_neo4j_pipeline

echo ""
echo "=== PHASE 5: Query (in Python) ==="
python3 << 'EOF'

# METHOD 1: Agent (Easiest)
from agents.orchestrator import run_agent

state = run_agent(
    query="What were NVIDIA's key risk factors in 2024?",
    chat_history=[],
    top_k=5
)

print("\n=== ANSWER ===")
print(state["answer"])

print("\n=== SOURCES ===")
for source in state["sources"]:
    print(f"  - {source}")

print("\n=== ROUTING ===")
print(f"Used: {state['tool_trace']}")

EOF

echo ""
echo "=== PHASE 6: Multi-Turn Conversation ==="
python3 << 'EOF'

from agents.orchestrator import run_agent

history = []

# Turn 1
print("\n[Turn 1] What is NVIDIA's data center revenue?")
result1 = run_agent(
    query="What is NVIDIA's data center revenue?",
    chat_history=history,
    top_k=5
)
print(result1["answer"])
history.append({"role": "user", "content": "What is NVIDIA's data center revenue?"})
history.append({"role": "assistant", "content": result1["answer"]})

# Turn 2 (agent remembers Turn 1)
print("\n[Turn 2] How much did that grow year-over-year?")
result2 = run_agent(
    query="How much did that grow year-over-year?",
    chat_history=history,
    top_k=5
)
print(result2["answer"])

EOF

echo ""
echo "=== PHASE 7: Direct Retrieval (No Agent) ==="
python3 << 'EOF'

from retrieval.hybrid_retriever import HybridRetriever
from openai import OpenAI

# Get chunks
retriever = HybridRetriever()
results = retriever.retrieve(
    "What are NVIDIA's main business risks?",
    top_k=5,
    hybrid=True
)

print(f"\n=== {len(results)} CHUNKS RETRIEVED ===")
for i, chunk in enumerate(results, 1):
    ticker = chunk.get("metadata", {}).get("ticker", "?")
    text = chunk.get("text", "")[:200]
    print(f"\n[{i}] {ticker}")
    print(f"    {text}...")

# Now call LLM manually
context = "\n\n".join([r.get("text", "") for r in results])

client = OpenAI(
    api_key="ignored",
    base_url="http://localhost:8000/v1"  # or Modal URL
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",
    messages=[
        {"role": "system", "content": "You are a financial analyst. Answer based only on the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What are NVIDIA's main business risks?"}
    ],
    temperature=0.2,
    max_tokens=500
)

print(f"\n=== LLM ANSWER ===")
print(response.choices[0].message.content)

EOF

echo ""
echo "=== PHASE 8: Just Get Raw Chunks (No LLM) ==="
python3 << 'EOF'

from retrieval.hybrid_retriever import HybridRetriever

r = HybridRetriever()
results = r.retrieve("What were NVIDIA's revenue trends?", top_k=3)

print(f"\n=== RAW CHUNKS ===")
for i, chunk in enumerate(results, 1):
    ticker = chunk.get("metadata", {}).get("ticker", "?")
    filing = chunk.get("metadata", {}).get("filing_type", "?")
    text = chunk.get("text", "")[:300]
    print(f"\n[{i}] {ticker} {filing}")
    print(f"    {text}...")

EOF

echo ""
echo "✅ All done! Run phases 5-8 anytime to query."
