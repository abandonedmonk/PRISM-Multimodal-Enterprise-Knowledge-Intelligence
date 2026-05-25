# PRISM — Multimodal Enterprise Knowledge Intelligence

SEC filing ingestion, Neo4j GraphRAG, hybrid retrieval, and multimodal AI — develop locally, deploy to Modal + EKS.

## Quick Navigation

- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Models](#models)
- [Pipeline Details](#pipeline-details)
- [Quick Terminal Commands](#quick-terminal-commands)
- [Complete Usage Guide: From Setup to Queries](#complete-usage-guide-from-setup-to-queries)
- [Query Methods](#query-methods)
- [All CLI Commands Reference](#all-cli-commands-reference)
- [Configuration & Provider Selection](#configuration--provider-selection)
- [Data Paths](#data-paths)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  SEC EDGAR    │────▶│  Ingestion   │────▶│   Neo4j +    │
│  Filings      │     │  Pipeline    │     │   Qdrant     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
          ┌─────────────┐    ┌──────────────┐    │
          │   LangGraph │◀──▶│  Retrieval   │◀──┘
          │   Agent     │    │  (Hybrid)    │
          └──────┬──────┘    └──────────────┘
                 │
          ┌──────▼───────┐
          │   FastAPI    │
          │   REST       │
          └──────────────┘
```

- **Ingestion** — download, clean, parse, chunk, table enrichment (Playwright PNG + VLM), entity extraction, graph construction, community reports, vector indexing
- **Retrieval** — Qdrant vector search, Neo4j graph search (local/global), hybrid merge, cross-encoder reranking
- **Agent** — LangGraph state machine: query routing, intelligent tool selection (hybrid/graph/web/vision), answer generation
- **Serving** — FastAPI REST; LLM via Modal serverless (Qwen 14B, Llama 11B Vision, BGE-small) or local vLLM

## Repository Layout

```
modal/               Modal serve scripts (LLM, Vision, Embedding, Rerank)
ingestion/
  core/              entity_extractor, graph_builder, community_detector, ...
  scripts/           CLI entrypoints (download, process, ingest, neo4j_pipeline)
retrieval/           hybrid_retriever, graph_retriever, reranker
agents/
  orchestrator.py    LangGraph agent: routing, tools, answer generation
  state.py           CorpMindState TypedDict definition
  tools/             retrieval, graph_traversal, web_search, vision
api/
  main.py            FastAPI app
  routes/            /chat, /documents, /voice endpoints
config.yaml          Non-secret defaults
config.py            Config loader (dotenv + yaml)
.env                 Secrets (never committed)
```

## Models

| Task | Model | Default Provider | GPU | Cost |
|------|-------|-----------------|-----|------|
| Entity extraction + community reports | `Qwen/Qwen2.5-14B-Instruct` | Modal (A100 40GB) or local vLLM | 1x A100 / any GPU with ≥28GB | ~$2.10/hr Modal |
| Table vision description | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Modal (A10) or local vLLM | 1x A10 / any GPU with ≥24GB | ~$1.10/hr Modal |
| Dense embeddings | `BAAI/bge-small-en-v1.5` | Modal (CPU) or in-process | CPU | ~$0.05/hr Modal |
| Reranking | `BAAI/bge-reranker-base-v2.0` | Modal (CPU) or in-process | CPU | ~$0.05/hr Modal |
| Sparse embeddings | `prithvida/Splade_PP_en_v1` | In-process (fastembed) | CPU | Free |

---

## Pipeline Details

### Ingestion Pipeline (Qdrant)

1. **Clean** — Strip XBRL/iXBRL markup, hidden elements, boilerplate
2. **Parse** — Partition HTML into typed elements (NarrativeText, Table, ListItem)
3. **Chunk** — Split into parent (3500 tok) / child (1024 tok) hierarchy
4. **Table enrichment** — Render tables as PNG via Playwright, describe with VLM (5 images/batch)
5. **Write JSONL** — Serialize chunks to `data/processed/*_chunks.jsonl`
6. **Embed** — Dense (BGE-small) + sparse (Splade) vectors
7. **Upsert** — Write to Qdrant `prism_filings` collection

### GraphRAG Pipeline (Neo4j)

1. **Entity extraction** — Qwen 14B, 10 chunks/batch, one-shot JSON prompt, checkpointed per-chunk to JSONL
2. **Graph build** — MERGE entities/relations into Neo4j, dedup by `name::type` canonical key
3. **Community detection** — Louvain via Neo4j GDS plugin
4. **Community reports** — Qwen 14B, one-shot JSON prompt, per-community checkpointing
5. **Vector index** — Embed entity summaries + community reports → Neo4j vector index

### Agent Orchestration (LangGraph)

1. **Query routing** — Classify user query into routes: `hybrid` (default), `graph_local` (entity-focused), `graph_global` (thematic), `web`, or `vision`
2. **Tool dispatch** — Route to appropriate retrieval tool(s): hybrid retriever, graph local/global search, web search, or vision analysis
3. **Context assembly** — Gather source chunks and structured context from selected tools
4. **LLM answer** — Pass context + chat history to Qwen 14B with system prompt; maintain conversation state
5. **Source tracking** — Return answer + original chunk/source metadata for transparency

---

## Quick Terminal Commands

**Copy-paste ready scripts** (in repo):
- `TLDR.sh` — One-liners for setup + querying (no commentary)
- `query_examples.py` — Six executable query examples (run: `python query_examples.py 1`)
- `QUICKSTART.sh` — Full annotated bash guide with all phases

Or read the detailed guide below.

---

## Complete Usage Guide: From Setup to Queries

**Fastest path**: follow this exact sequence. Estimated time: 30–45 minutes (+ 2–3 hours for first ingestion if using local vLLM).

### Phase 1: One-Time Setup (10 min)

```bash
# Clone and navigate to repo
cd /path/to/PRISM

# Set up Python environment
python3.10 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Copy config template
cp .env.example .env
# Edit .env — set NEO4J_PASSWORD and any API keys if using Fireworks/Groq
```

### Phase 2: Start Data Services (5 min)

Open a terminal and leave running:

```bash
docker-compose up -d
docker-compose ps  # both qdrant and neo4j should show "healthy" in ~30 sec
```

If needed, view logs:
```bash
docker-compose logs -f neo4j
docker-compose logs -f qdrant
```

### Phase 3: Deploy LLM Services (Choose One)

#### Option A: Modal (Serverless GPU — Recommended, $0.30/hr or free tier)

```bash
# Login to Modal
modal login

# Deploy all four services (each ~3–5 min, can be parallel)
modal deploy modal/llm_serve.py &
modal deploy modal/vision_serve.py &
modal deploy modal/embed_serve.py &
modal deploy modal/rerank_serve.py &
wait

# Copy the returned URLs into config.yaml
# Example:
#   llm.base_url = "https://user--prism-llm-serve-serve.modal.run/v1"
#   vision.base_url = "https://user--prism-vision-serve-serve.modal.run/v1"
#   embedding.base_url = "https://user--prism-embed-serve-serve.modal.run"
#   rerank.base_url = "https://user--prism-rerank-serve-serve.modal.run"

# Verify deployment
curl https://user--prism-llm-serve-serve.modal.run/v1/models

# Warm startup (invoke services once to wake them up)
modal invoke modal/llm_serve.py
modal invoke modal/vision_serve.py
modal invoke modal/embed_serve.py
modal invoke modal/rerank_serve.py
```

#### Option B: Local vLLM (GPU-required, free, ~3 min to load models)

Requires GPU with ≥40GB VRAM. Run in separate terminals:

Terminal 1:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Terminal 2:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt image=5
```

(config.yaml defaults already point to `localhost:8000/v1` and `localhost:8001/v1`)

#### Option C: Cloud APIs (No GPU needed)

Edit `config.yaml`:
```yaml
llm:
  base_url: "https://api.fireworks.ai/inference/v1"
  model: "accounts/fireworks/models/gpt-oss-120b"

vision:
  base_url: "https://api.groq.com/openai/v1"
  model: "meta-llama/llama-4-scout-17b-16e-instruct"
```

Set secrets in `.env`:
```bash
FIREWORKS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### Phase 4: Prepare Data (5-30 min depending on setup)

#### Download sample SEC filings:
```bash
python -m ingestion.scripts.download_filings --tickers NVDA AAPL
```

This downloads 10-K/10-Q filings for NVIDIA and Apple into `data/raw/`.

#### Process filings into chunks:
```bash
python -m ingestion.scripts.process_all_filings --force
```

Produces `data/processed/*_chunks.jsonl` and `data/processed/tables/*.png`.

#### Ingest chunks into Qdrant (dense + sparse vectors):
```bash
python -m ingestion.scripts.ingest --recreate
```

#### Build Neo4j GraphRAG index (entity extraction + graph + community reports):
```bash
python -m ingestion.scripts.run_neo4j_pipeline
```

*Tip: If this times out, use `--skip-extraction` to reuse checkpoint:*
```bash
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction
```

### Phase 5: Query the LLM (Main Interface)

#### Method 1: Via Agent (Recommended — Conversational, Auto-Routing)

**In Python REPL or script:**

```python
from agents.orchestrator import run_agent

# Single query
state = run_agent(
    query="What were NVIDIA's key risk factors in 2024?",
    chat_history=[],
    top_k=5
)

print("Answer:")
print(state["answer"])
print("\nSources:")
for source in state["sources"]:
    print(f"  - {source}")
print("\nTool trace:")
print(state["tool_trace"])  # ["route", "hybrid", "answer"] or similar
```

**Multi-turn conversation:**

```python
from agents.orchestrator import run_agent

history = []

# Turn 1
result1 = run_agent(
    query="What is NVIDIA's data center revenue?",
    chat_history=history,
    top_k=5
)
history.append({"role": "user", "content": "What is NVIDIA's data center revenue?"})
history.append({"role": "assistant", "content": result1["answer"]})

# Turn 2 (agent sees turn 1 context)
result2 = run_agent(
    query="How much did that grow year-over-year?",
    chat_history=history,
    top_k=5
)
print(result2["answer"])
```

#### Method 2: Direct Hybrid Retrieval + Manual LLM Call

If you want to skip the agent and handle retrieval + LLM yourself:

```python
from retrieval.hybrid_retriever import HybridRetriever
from openai import OpenAI
import os

# Retrieve context
retriever = HybridRetriever()
results = retriever.retrieve(
    "What are NVIDIA's main business risks?",
    top_k=5,
    hybrid=True
)

# Assemble context
context = "\n\n".join([r.get("text", "") for r in results])

# Call LLM directly
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "not-needed"),
    base_url="http://localhost:8000/v1"  # or Modal URL
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful analyst. Answer based on the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What are NVIDIA's main business risks?"}
    ],
    temperature=0.2,
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Method 3: Direct Retrieval Without LLM (Just Get Context)

```python
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.graph_retriever import GraphRetriever

# Hybrid search (Qdrant + Neo4j)
retriever = HybridRetriever()
chunks = retriever.retrieve("What were NVIDIA's revenue trends?", top_k=5)
for i, chunk in enumerate(chunks, 1):
    print(f"\n[Chunk {i}] {chunk.get('ticker', '')} {chunk.get('filing_type', '')}")
    print(chunk.get("text", "")[:300] + "...")

# Or: graph-only local search (entity-focused)
graph = GraphRetriever()
local = graph.local_search(
    entity_keys=["NVIDIA"],
    top_k=5,
    hop=2
)
print("Related entities:", [e["name"] for e in local["entities"]])
print("Context:", local["context_text"])
graph.close()

# Or: graph-only global search (thematic)
import numpy as np
embedding = np.random.randn(384).tolist()  # or use real embedding
global_ctx = graph.global_search(query_embedding=embedding, top_k=3)
print("Top communities:", [c["title"] for c in global_ctx["communities"]])
```

#### Method 4: Call LLM for Ingestion Tasks (Entity Extraction, Reports)

These happen automatically in `run_neo4j_pipeline`, but you can call them manually:

```python
from ingestion.core.entity_extractor import extract_entities_from_chunks

chunks = [
    {
        "chunk_id": "001",
        "text": "Apple Inc. reported Q4 revenue of $123B, up 5% YoY. ..."
    }
]

# Extract entities + relations
result = extract_entities_from_chunks(chunks, batch_size=1)
print(result)  # {"chunk_id": "001", "entities": [...], "relations": [...]}
```

### Phase 6: Running Evaluations (Optional)

```bash
# Run RAGAS evaluation on a test set
python eval/evaluate.py --test-set eval/test_set.json --output eval/results.json

# Check results
cat eval/results.json | jq '.metrics'
```

---

## Query Methods

### Agent Routing Rules

The agent automatically routes based on query keywords:

| Keywords | Route | Retriever |
|----------|-------|----------|
| "trend", "theme", "global", "overall" | `graph_global` | Community reports + member chunks |
| "relationship", "related", "connected", "impact" | `graph_local` | Entity-neighborhood traversal |
| "latest", "news", "web", "online" | `web` | External web search (placeholder) |
| "chart", "table", "image", "figure" | `vision` | VLM table/image description (placeholder) |
| (default) | `hybrid` | Qdrant + Neo4j + reranking |

### Query / Agent Cheat Sheet

```python
# Via Agent (recommended for conversational UX)
from agents.orchestrator import run_agent

state = run_agent(
    query="What were NVIDIA's key risk factors?",
    chat_history=[],          # accumulate for multi-turn
    top_k=5,
    use_web=False,            # enable external search if needed
    use_vision=False,         # enable chart analysis if needed
)
print(state["answer"])
print(state["sources"])        # original chunks + metadata
print(state["tool_trace"])     # ["route", "hybrid", "answer"]

# Direct Retriever (lower-level, if you want to skip agent)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.graph_retriever import GraphRetriever

# Hybrid (Qdrant + Neo4j + reranking)
r = HybridRetriever()
results = r.retrieve("What were NVIDIA's key risk factors?", top_k=5)

# Graph-only local (entity-focused)
g = GraphRetriever()
local = g.local_search(query_embedding=..., top_k=5, hop=2)

# Graph-only global (thematic)
global_ctx = g.global_search(query_embedding=..., top_k=5)
```

---

## All CLI Commands Reference

### Modal

```bash
# Deploy
modal deploy modal/llm_serve.py          # Qwen 14B on A100
modal deploy modal/vision_serve.py        # Llama 11B Vision on A10
modal deploy modal/embed_serve.py         # BGE-small on CPU
modal deploy modal/rerank_serve.py        # BGE reranker on CPU

# Test (spins up a replica, runs health check + sample request)
modal run modal/llm_serve.py
modal run modal/vision_serve.py
modal run modal/embed_serve.py
modal run modal/rerank_serve.py
```

### Docker

```bash
docker-compose up -d                      # Start Neo4j + Qdrant
docker-compose ps                         # Check health
docker-compose down                       # Stop (data preserved in volumes)
docker-compose down -v                    # Stop and DELETE volumes
```

### Data Download

```bash
# Download default tickers (NVDA, AMD, INTC, AAPL, MSFT, GOOGL, META, AMZN, TSLA, JPM)
python -m ingestion.scripts.download_filings

# Download specific tickers
python -m ingestion.scripts.download_filings --tickers NVDA AAPL

# Dry run (list what would be downloaded)
python -m ingestion.scripts.download_filings --dry-run
```

### Ingestion

```bash
# Process all raw filings: clean → parse → chunk → JSONL
python -m ingestion.scripts.process_all_filings            # skip already-processed
python -m ingestion.scripts.process_all_filings --force    # reprocess everything

# Ingest chunks into Qdrant
python -m ingestion.scripts.ingest                         # add to existing collection
python -m ingestion.scripts.ingest --recreate              # drop and recreate collection

# Build Neo4j GraphRAG index (full pipeline)
python -m ingestion.scripts.run_neo4j_pipeline

# Build Neo4j GraphRAG index (skip extraction, use checkpoint)
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction

# Build Neo4j GraphRAG index (dry run, preview only)
python -m ingestion.scripts.run_neo4j_pipeline --dry-run

# Summarize table chunks with LLM for better embeddings
python -m ingestion.scripts.summarize_tables
python -m ingestion.scripts.summarize_tables --dry-run

# Run full pipeline (download → process → ingest → graph)
python -m ingestion.scripts.pipeline
```

---

## Configuration & Provider Selection

### Configuration Files

| File | Contents | Committed? |
|------|----------|------------|
| `config.yaml` | Non-secret defaults (endpoints, models, paths, dimensions) | Yes |
| `config.py` | Loads `.env` + `config.yaml`, exports all config vars | Yes |
| `.env` | Secrets only (API keys, passwords) | **No** |
| `.env.example` | Template showing which secrets to fill | Yes |

### Switching Providers

All switching happens in `config.yaml` — no code changes needed:

```yaml
# Option A: Modal (serverless GPU)
llm:
  base_url: "https://<ws>--prism-llm-serve-serve.modal.run/v1"

# Option B: Local vLLM
llm:
  base_url: "http://localhost:8000/v1"

# Option C: Fireworks (cloud API)
llm:
  base_url: "https://api.fireworks.ai/inference/v1"
```

Same pattern for `vision:` and `embedding:` sections.

## Data Paths

All paths are static defaults from `config.yaml`:

| Path | Contents |
|------|----------|
| `data/raw/*.htm` | Raw SEC EDGAR filings |
| `data/processed/*_chunks.jsonl` | Chunked filings (one JSON object per line) |
| `data/processed/tables/*.png` | Rendered table images |
| `data/graphs/graph_extractions_checkpoint.json` | Entity extraction checkpoint (JSONL) |
| `data/graphs/community_reports_checkpoint.json` | Community report checkpoint (JSONL) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ingestion'` | Run from repo root with active venv |
| `No files matching '*.htm'` | Run `python -m ingestion.scripts.download_filings` or place HTM files in `data/raw/` |
| `Qdrant connection refused` | Start services: `docker-compose up -d` |
| `Neo4j connection refused` | Start services: `docker-compose up -d` |
| `GDS plugin not available` | Ensure Neo4j GDS plugin is mounted in `neo4j_plugins/` |
| `vLLM OOM on Qwen 14B` | Use a GPU with ≥28GB VRAM, or switch to Modal/Fireworks in config |
| `vLLM OOM on Llama 11B Vision` | Use a GPU with ≥24GB VRAM, or switch to Modal/Groq in config |
| `Modal deploy fails` | Run `modal login` first, ensure you have Modal credits |
| `Entity extraction returns empty JSON` | LLM endpoint is down — check `config.yaml` `llm.base_url` points to a running server |
| `VLM descriptions all empty` | Vision endpoint is down — check `config.yaml` `vision.base_url` |

## License

MIT
