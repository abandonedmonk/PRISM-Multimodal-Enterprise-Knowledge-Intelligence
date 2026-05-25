# PRISM Ingestion Pipeline

End-to-end SEC EDGAR filing ingestion: download, clean, parse, chunk, enrich, and write retrieval indexes.

This package is the offline/write path. It prepares filing chunks for Qdrant and builds the Neo4j GraphRAG index used later by the `retrieval/` package.

## Quick Start

```bash
# 1. Configure secrets
cp .env.example .env  # fill in NEO4J_PASSWORD at minimum

# 2. Deploy Modal endpoints (or start local vLLM — see main README)
modal deploy modal/llm_serve.py
modal deploy modal/vision_serve.py
modal deploy modal/embed_serve.py

# 3. Install deps
pip install -r requirements.txt
playwright install chromium

# 4. Start data services
docker-compose up -d

# 5. Download filings (optional — skip if you have .htm files in data/raw/)
python -m ingestion.scripts.download_filings

# 6. Process filings: clean -> parse -> chunk -> JSONL
python -m ingestion.scripts.process_all_filings

# 7. Ingest to Qdrant
python -m ingestion.scripts.ingest --recreate

# 8. Build Neo4j GraphRAG index
python -m ingestion.scripts.run_neo4j_pipeline
```

## CLI Commands

### Download

```bash
# Default tickers (NVDA, AMD, INTC, AAPL, MSFT, GOOGL, META, AMZN, TSLA, JPM)
python -m ingestion.scripts.download_filings

# Specific tickers
python -m ingestion.scripts.download_filings --tickers NVDA AAPL

# Dry run
python -m ingestion.scripts.download_filings --dry-run
```

### Process

```bash
python -m ingestion.scripts.process_all_filings            # skip already-processed
python -m ingestion.scripts.process_all_filings --force    # reprocess everything
```

### Ingest (Qdrant)

```bash
python -m ingestion.scripts.ingest                         # add to existing collection
python -m ingestion.scripts.ingest --recreate              # drop and recreate collection
```

### Neo4j GraphRAG

```bash
python -m ingestion.scripts.run_neo4j_pipeline             # full pipeline
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction  # reuse extraction checkpoint
python -m ingestion.scripts.run_neo4j_pipeline --dry-run         # preview only
```

### Table Summarization

```bash
python -m ingestion.scripts.summarize_tables               # summarize table chunks with LLM
python -m ingestion.scripts.summarize_tables --dry-run     # preview only
```

### Full Pipeline (download → process → ingest → graph)

```bash
python -m ingestion.scripts.pipeline
```

## Pipeline Steps

### 1. Download (Optional)

- **Module:** `scripts/download_filings.py`
- Fetches 10-K/10-Q HTM files from SEC EDGAR API
- Rate-limited (0.15s between requests)
- Output: `data/raw/TICKER_YEAR_FORM.htm`

### 2. Clean (Automatic)

- **Module:** `core/cleaner.py`
- Strips XBRL/iXBRL markup, hidden elements, boilerplate
- Outputs `*_clean.htm` files (80-90% of original size)

### 3. Parse

- **Module:** `core/parser.py`
- Partitions HTML into typed elements using `unstructured`
- Detects 16+ 10-K/10-Q sections (Item 1, 1A, 7, 8, etc.)
- Annotates elements with section and anchor IDs

### 4. Chunk

- **Module:** `core/chunker.py`
- Splits into parent (3500 tok) / child (1024 tok) hierarchy
- Groups by 10-K section, skips boilerplate sections
- Extracts table markdown using pandas
- Renders tables as PNG via Playwright, describes with VLM (5 images/batch)

### 5. Write Processed (Automatic)

- **Module:** `scripts/process_all_filings.py`
- Orchestrates steps 2-4, writes chunks to JSONL

### 6. Ingest (Qdrant)

- **Module:** `scripts/ingest.py`
- Embeds chunks with dense (BGE-small) + sparse (Splade) vectors
- Upserts to Qdrant `prism_filings` collection

### 7. Build GraphRAG Index (Neo4j)

- **Module:** `scripts/run_neo4j_pipeline.py`
- 5-step pipeline: extract entities → build graph → detect communities → generate reports → index vectors

## Core Modules

| Module | Purpose |
|--------|---------|
| `cleaner.py` | HTML cleaning, XBRL removal, TOC extraction |
| `parser.py` | Element partitioning, section detection |
| `chunker.py` | Text splitting, chunk generation, table extraction |
| `metadata.py` | Filename parsing, CIK/ticker lookup, source URLs |
| `table_renderer.py` | Playwright-based table-to-PNG renderer |
| `vision_extractor.py` | VLM table image description (5/batch, auto rate-limit) |
| `entity_extractor.py` | Qwen 14B entity/relation extraction (10 chunks/batch, one-shot JSON) |
| `graph_builder.py` | Neo4j graph construction with MERGE deduplication |
| `community_detector.py` | Neo4j GDS Louvain community detection |
| `community_reports.py` | Qwen 14B community report generation (one-shot JSON) |
| `vector_indexer.py` | Neo4j vector embeddings (remote Modal or local fastembed) |

## Configuration

Ingestion uses `config.yaml` for non-secret defaults and `.env` for secrets. All modules import from `config.py`.

Required env vars (depends on provider):
- `NEO4J_PASSWORD` — always required
- `FIREWORKS_API_KEY` — only if using Fireworks fallback
- `GROQ_API_KEY` — only if using Groq vision fallback
- `LLM_API_KEY` — optional, defaults to "not-needed" for local vLLM / Modal

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ingestion'` | Run from repo root with active venv |
| `No files matching '*.htm'` | Run `download_filings.py` or place HTM files in `data/raw/` |
| `Qdrant connection refused` | Start services: `docker-compose up -d` |
| `GDS plugin not available` | Ensure Neo4j GDS plugin is mounted in `neo4j_plugins/` |
| `Entity extraction returns empty JSON` | LLM endpoint down — check `config.yaml` `llm.base_url` |
| `VLM descriptions all empty` | Vision endpoint down — check `config.yaml` `vision.base_url` |
