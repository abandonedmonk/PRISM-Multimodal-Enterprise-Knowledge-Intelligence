import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Centralized configuration.
# Non-secret values come from config.yaml; secrets stay in .env (env vars).

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

_DEFAULTS = {
    "llm": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-14B-Instruct",
    },
    "vision": {
        "base_url": "http://localhost:8001/v1",
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    },
    "embedding": {
        "base_url": "",
        "model": "BAAI/bge-small-en-v1.5",
    },
    "rerank": {
        "base_url": "",
        "model": "BAAI/bge-reranker-base-v2.0",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/gpt-4o-mini",
    },
    "groq": {"base_url": "https://api.groq.com/openai/v1"},
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "entity_extraction_model": "Qwen/Qwen2.5-14B-Instruct",
        "community_report_model": "Qwen/Qwen2.5-14B-Instruct",
        "vector_index_name": "entity_embeddings",
        "vector_dimensions": 384,
        "vector_similarity": "cosine",
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collection": "prism_filings",
    },
    "models": {
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "table": {
        "images_dir": "data/processed/tables",
        "render_width": 1200,
    },
    "services": {
        "embedding_service_url": "http://localhost:8001",
        "reranker_service_url": "http://localhost:8002",
        "llm_endpoint": "https://api.fireworks.ai/inference/v1",
    },
    "langfuse": {"host": "http://localhost:3000"},
    "aws": {
        "region": "ap-south-1",
        "s3_docs_bucket": "prism-docs-local",
        "s3_snapshots_bucket": "prism-snapshots-local",
    },
    "ingestion": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "graph_dir": "data/graphs",
        "graph_batch_size": 8,
        "graph_concurrency": 4,
        "filings_glob": "*.htm",
    },
}


def _load_config_yaml() -> dict:
    if not CONFIG_YAML_PATH.exists():
        return {}
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg(path: list[str], default):
    data = _CONFIG
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


_CONFIG = _load_config_yaml()

# Secrets (env only)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Non-secret config (config.yaml)
LLM_BASE_URL = _cfg(["llm", "base_url"], _DEFAULTS["llm"]["base_url"])
LLM_MODEL = _cfg(["llm", "model"], _DEFAULTS["llm"]["model"])

VISION_BASE_URL = _cfg(["vision", "base_url"], _DEFAULTS["vision"]["base_url"])
VISION_MODEL = _cfg(["vision", "model"], _DEFAULTS["vision"]["model"])

EMBEDDING_BASE_URL = _cfg(["embedding", "base_url"], _DEFAULTS["embedding"]["base_url"])
EMBEDDING_MODEL = _cfg(["embedding", "model"], _DEFAULTS["embedding"]["model"])

RERANKER_BASE_URL = _cfg(["rerank", "base_url"], _DEFAULTS["rerank"]["base_url"])
RERANKER_MODEL = _cfg(["rerank", "model"], _DEFAULTS["rerank"]["model"])

FIREWORKS_BASE_URL = _cfg(["fireworks", "base_url"], _DEFAULTS["fireworks"]["base_url"])
FIREWORKS_MODEL = _cfg(["fireworks", "model"], _DEFAULTS["fireworks"]["model"])

GROQ_BASE_URL = _cfg(["groq", "base_url"], _DEFAULTS["groq"]["base_url"])

NEO4J_URI = _cfg(["neo4j", "uri"], _DEFAULTS["neo4j"]["uri"])
NEO4J_USER = _cfg(["neo4j", "user"], _DEFAULTS["neo4j"]["user"])
NEO4J_ENTITY_EXTRACTION_MODEL = _cfg(
    ["neo4j", "entity_extraction_model"],
    _DEFAULTS["neo4j"]["entity_extraction_model"],
)
NEO4J_COMMUNITY_REPORT_MODEL = _cfg(
    ["neo4j", "community_report_model"],
    _DEFAULTS["neo4j"]["community_report_model"],
)
NEO4J_VECTOR_INDEX_NAME = _cfg(
    ["neo4j", "vector_index_name"],
    _DEFAULTS["neo4j"]["vector_index_name"],
)
NEO4J_VECTOR_DIMENSIONS = int(
    _cfg(["neo4j", "vector_dimensions"], _DEFAULTS["neo4j"]["vector_dimensions"])
)
NEO4J_VECTOR_SIMILARITY = _cfg(
    ["neo4j", "vector_similarity"],
    _DEFAULTS["neo4j"]["vector_similarity"],
)

QDRANT_HOST = _cfg(["qdrant", "host"], _DEFAULTS["qdrant"]["host"])
QDRANT_PORT = int(_cfg(["qdrant", "port"], _DEFAULTS["qdrant"]["port"]))
QDRANT_COLLECTION = _cfg(["qdrant", "collection"], _DEFAULTS["qdrant"]["collection"])

TABLE_IMAGES_DIR = _cfg(["table", "images_dir"], _DEFAULTS["table"]["images_dir"])
TABLE_RENDER_WIDTH = int(
    _cfg(["table", "render_width"], _DEFAULTS["table"]["render_width"])
)

EMBEDDING_SERVICE_URL = _cfg(
    ["services", "embedding_service_url"],
    _DEFAULTS["services"]["embedding_service_url"],
)
RERANKER_SERVICE_URL = _cfg(
    ["services", "reranker_service_url"],
    _DEFAULTS["services"]["reranker_service_url"],
)

LLM_ENDPOINT = _cfg(["services", "llm_endpoint"], _DEFAULTS["services"]["llm_endpoint"])

LANGFUSE_HOST = _cfg(["langfuse", "host"], _DEFAULTS["langfuse"]["host"])

AWS_REGION = _cfg(["aws", "region"], _DEFAULTS["aws"]["region"])
S3_DOCS_BUCKET = _cfg(["aws", "s3_docs_bucket"], _DEFAULTS["aws"]["s3_docs_bucket"])
S3_SNAPSHOTS_BUCKET = _cfg(
    ["aws", "s3_snapshots_bucket"],
    _DEFAULTS["aws"]["s3_snapshots_bucket"],
)

INGESTION_RAW_DIR = _cfg(
    ["ingestion", "raw_dir"], _DEFAULTS["ingestion"]["raw_dir"]
)
INGESTION_PROCESSED_DIR = _cfg(
    ["ingestion", "processed_dir"], _DEFAULTS["ingestion"]["processed_dir"]
)
INGESTION_GRAPH_DIR = _cfg(
    ["ingestion", "graph_dir"], _DEFAULTS["ingestion"]["graph_dir"]
)
INGESTION_GRAPH_BATCH_SIZE = int(
    _cfg(["ingestion", "graph_batch_size"], _DEFAULTS["ingestion"]["graph_batch_size"])
)
INGESTION_GRAPH_CONCURRENCY = int(
    _cfg(["ingestion", "graph_concurrency"], _DEFAULTS["ingestion"]["graph_concurrency"])
)
INGESTION_FILINGS_GLOB = _cfg(
    ["ingestion", "filings_glob"], _DEFAULTS["ingestion"]["filings_glob"]
)
