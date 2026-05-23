# PRISM Ingestion Pipeline

End-to-end SEC EDGAR filing ingestion: download, clean, parse, chunk, and embed into vector database.

## Quick Start

**Minimal example: process a single filing into chunks (JSONL)**

```bash
# 1. Activate virtual environment (if not already activated)
source ../.venv/bin/activate

# 2. Download filings (optional — sample data already in data/raw/)
python -m ingestion.scripts.download_filings --tickers NVDA --dry-run

# 3. Process filings: clean → parse → chunk → JSONL
python -m ingestion.scripts.process_all_filings

# 4. Output: chunks written to data/processed/{filename}_chunks.jsonl
```

## Pipeline Steps

### 1. **Download** (Optional)
- **Module:** `scripts/download_filings.py`
- **Purpose:** Fetch 10-K and 10-Q filings from SEC EDGAR API
- **Output:** Raw HTM files in `data/raw/`
- **Command:**
  ```bash
  python -m ingestion.scripts.download_filings \
    --tickers AAPL NVDA MSFT \
    --output-dir data/raw
  ```

### 2. **Clean** (Automatic)
- **Module:** `core/cleaner.py`
- **Purpose:** Strip XBRL/iXBRL markup, hidden elements, boilerplate
- **Output:** `*_clean.htm` files (80-90% of original size)
- **Details:**
  - Removes `<ix:*>` tags
  - Removes display:none elements
  - Extracts table of contents (TOC) for section identification

### 3. **Parse**
- **Module:** `core/parser.py`
- **Purpose:** Partition HTML into typed elements using `unstructured`
- **Output:** List of `ParsedElement` objects with section tags
- **Element types:** Text, NarrativeText, Table, ListItem, Image
- **Details:**
  - Detects 16+ 10-K/10-Q sections (Item 1, 1A, 7, 8, etc.)
  - Filters boilerplate (~50 words minimum)
  - Annotates elements with section and anchor IDs

### 4. **Chunk**
- **Module:** `core/chunker.py`
- **Purpose:** Split elements into parent/child chunks for RAG
- **Output:** List of `Chunk` objects (text, metadata, section, CIK, etc.)
- **Parameters:**
  - Parent chunks: 2000 tokens (200 overlap)
  - Child chunks: 512 tokens (100 overlap)
  - **Recommendation for SEC:** Consider increasing to 1024 child tokens for better financial context retention
- **Details:**
  - Groups by 10-K section (skips irrelevant sections: item4, item6, item10, etc.)
  - Extracts table markdown using pandas
  - Generates unique chunk IDs and parent/child relationships
  - Builds source URLs pointing to SEC EDGAR

### 5. **Write Processed** (Automatic)
- **Module:** `scripts/process_all_filings.py`
- **Purpose:** Orchestrate steps 2–4 and write chunks to JSONL
- **Output:** `data/processed/{filename}_chunks.jsonl` (one chunk per line)
- **Command:**
  ```bash
  python -m ingestion.scripts.process_all_filings \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --force  # Re-process even if output exists
  ```

### 6. **Summarize Tables** (Optional)
- **Module:** `scripts/summarize_tables.py`
- **Purpose:** Use LLM to summarize table markdown (improves embeddings)
- **Requires:** `FIREWORKS_API_KEY` environment variable
- **Command:**
  ```bash
  export FIREWORKS_API_KEY=your_key_here
  python -m ingestion.scripts.summarize_tables \
    --processed-dir data/processed \
    --model accounts/fireworks/models/gpt-4o-mini
  ```

### 7. **Ingest** (Vector DB)
- **Module:** `scripts/ingest.py`
- **Purpose:** Embed chunks with dense + sparse vectors, upsert to Qdrant
- **Requires:** Qdrant running (see `config.yaml`)
- **Output:** Vectors in Qdrant collection (e.g., `prism_filings`)
- **Command:**
  ```bash
  python -m ingestion.scripts.ingest \
    --processed-dir data/processed \
    --recreate  # Delete and recreate collection
  ```

## File Structure

### Core Modules (`core/`)
- **`cleaner.py`** — HTML cleaning, XBRL removal, TOC extraction
- **`parser.py`** — Element partitioning, section detection
- **`chunker.py`** — Text splitting, chunk generation, table extraction
- **`metadata.py`** — Filename parsing, CIK/ticker lookup, source URLs
- **`entity_extractor.py`** — Placeholder for NER pipeline
- **`vision_extractor.py`** — Placeholder for visual document analysis

### Scripts (`scripts/`)
- **`download_filings.py`** — SEC EDGAR API client
- **`process_all_filings.py`** — Main orchestration (clean → parse → chunk → JSONL)
- **`summarize_tables.py`** — LLM-based table summarization
- **`ingest.py`** — Qdrant vector ingestion
- **`pipeline.py`** — CLI entrypoint

### Dataset (`dataset/`)
- **`convert.py`** — Load/save chunks to JSONL; Hugging Face export (placeholder)
- **`schema.py`** — `FilingChunk` dataclass definition

### Tests (`tests/`)
- **`test_cleaner.py`** — Demo cleaner on sample filing
- **`test_parser.py`** — Demo parser and element counts
- **`test_e2e.py`** — Complete pipeline: clean → parse → chunk → JSONL

### Root Wrappers
- **`chunking.py`** — Convenience re-export of `Chunk` and `chunk_filing`

## Configuration

### `config.yaml` (in project root)
```yaml
qdrant:
  host: localhost
  port: 6333
  collection: prism_filings
  dense_dim: 384

models:
  dense_embedding: BAAI/bge-small-en-v1.5
  sparse_embedding: prithvida/Splade_PP_en_v1
```

If not present, `ingest.py` uses built-in defaults.

## End-to-End Example

From the repository root:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Process sample filings (already in data/raw)
python -m ingestion.scripts.process_all_filings

# 3. Check output
ls -lh data/processed/
# → NVDA_2024_10K_chunks.jsonl, AAPL_2024_10K_chunks.jsonl, etc.

# 4. (Optional) Summarize tables with LLM
export FIREWORKS_API_KEY=...
python -m ingestion.scripts.summarize_tables

# 5. (Optional) Ingest into Qdrant
python -m ingestion.scripts.ingest --recreate
```

## Data Paths

- **Raw filings:** `data/raw/*.htm`
- **Processed chunks:** `data/processed/*_chunks.jsonl`
- **Cleaned intermediate:** `data/raw/*_clean.htm` (temporary)

All processed outputs go to the repository `data/processed/` folder (see `ingestion/dataset/convert.py` for default path logic).

## Testing

Run individual demos to validate each step:

```bash
# Test cleaner
python ingestion/tests/test_cleaner.py --file data/raw/NVDA_2024_10K.htm

# Test parser
python ingestion/tests/test_parser.py --file data/raw/NVDA_2024_10K.htm

# Test end-to-end
python ingestion/tests/test_e2e.py --file data/raw/NVDA_2024_10K.htm --out data/processed/test_chunks.jsonl
```

## Dependencies

- `beautifulsoup4`, `lxml` — HTML parsing
- `unstructured` — Element partitioning
- `langchain-text-splitters` — Text chunking
- `pandas` — Table parsing
- `requests` — HTTP (download)
- `pyyaml` — Config parsing
- `qdrant-client` — Vector DB client
- `fastembed` — Dense/sparse embeddings
- `openai` — LLM API (optional, for table summarization)

## Output Schema

Each line in `*_chunks.jsonl` is a JSON object:

```json
{
  "text": "Block of text (up to 512 tokens)",
  "parent_text": "Full section text (up to 2000 tokens)",
  "table_markdown": "Markdown representation of table (if element_type='Table')",
  "company": "NVIDIA Corporation",
  "ticker": "NVDA",
  "cik": "0001045810",
  "year": 2024,
  "quarter": null,
  "filing_type": "10-K",
  "accession_number": "",
  "section": "Management's Discussion and Analysis",
  "anchor_id": "item7",
  "element_type": "NarrativeText",
  "chunk_index": 42,
  "parent_chunk_id": "uuid-...",
  "chunk_id": "uuid-...",
  "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/",
  "htm_filename": "NVDA_2024_10K.htm"
}
```

## Next Steps

1. **Increase chunk sizes for SEC:** Update `core/chunker.py` `parent_splitter` and `child_splitter` (consider 3500 and 1024 tokens respectively for financial context).
2. **Run full pipeline:** Process all filings and ingest into Qdrant.
3. **Tune embeddings:** Experiment with different dense/sparse models in `config.yaml`.
4. **Implement table summarization:** Wire LLM API key and run `summarize_tables.py`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ingestion'` | Ensure you're running from repo root and `.venv/bin/python` is active. |
| `No files matching '*.htm' in data/raw` | Run `download_filings.py` or place sample HTM files in `data/raw/`. |
| `Qdrant connection refused` | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` or adjust `config.yaml`. |
| `FIREWORKS_API_KEY not set` | Export key: `export FIREWORKS_API_KEY=...` before running `summarize_tables.py`. |

---

**Questions?** See design docs in `docs/` or check individual module docstrings.
