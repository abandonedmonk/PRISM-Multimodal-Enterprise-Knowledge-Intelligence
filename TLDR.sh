#!/bin/bash
# TLDR: One-Time Setup + Quick Query
# Copy-paste each block in order

# ===== SETUP (One-Time) =====
cd /path/to/PRISM
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && playwright install chromium
cp .env.example .env
# ^ Edit .env: set NEO4J_PASSWORD

# Start services (Terminal 1, leave running)
docker-compose up -d && docker-compose ps

# Deploy LLM (Terminal 2)
modal login
modal deploy modal/llm_serve.py & modal deploy modal/vision_serve.py & modal deploy modal/embed_serve.py & modal deploy modal/rerank_serve.py & wait

# ===== DATA (One-Time) =====
python -m ingestion.scripts.download_filings --tickers NVDA AAPL
python -m ingestion.scripts.process_all_filings --force
python -m ingestion.scripts.ingest --recreate
python -m ingestion.scripts.run_neo4j_pipeline

# ===== QUERY (Anytime) =====
python query_examples.py 1     # Agent (easiest)
python query_examples.py 2     # Multi-turn
python query_examples.py 3     # Manual LLM
python query_examples.py all   # All examples

# ===== INTERACTIVE PYTHON =====
python3
>>> from agents.orchestrator import run_agent
>>> state = run_agent(query="What were NVIDIA's risk factors?", chat_history=[], top_k=5)
>>> print(state["answer"])
