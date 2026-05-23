# PRISM — Multimodal Enterprise Knowledge Intelligence

PRISM is an extensible platform for building enterprise-grade multimodal knowledge intelligence: ingestion, retrieval, reasoning, and multimodal (text, vision, voice) interfaces that connect to private data stores and production inference pipelines.

Key goals
- Provide secure, auditable access to organization knowledge across documents, images, and audio
- Combine retrieval (hybrid retrieval + graph augmentation) with large multimodal agents
- Offer production-ready deployment options (Docker Compose, Kubernetes, Terraform)
- Support fine-tuning and evaluation workflows for model iteration

High-level architecture
- Ingestion: chunking, entity extraction, vision/audio feature extraction (folder: `ingestion/`)
- Retrieval: hybrid retrievers, graph-based retrieval and reranking (folder: `retrieval/`)
- Agents & orchestration: multi-agent orchestration, tools for search, vision, and web retrieval (folder: `agents/`)
- API: REST endpoints for chat, documents, and voice interfaces (folder: `api/`)
- Modal serving: components for vision and vLLM inference (folder: `modal/`)
- Evaluation & fine-tuning: datasets, training and evaluation scripts (folders: `eval/`, `fine_tuning/`)
- Infrastructure: Kubernetes manifests and Terraform modules for infrastructure provisioning (folders: `k8s/`, `terraform/`)

Repository layout
- `api/` — API server, routes and configuration
- `agents/` — orchestrator, agent state and helper tools
- `ingestion/` — document chunking and extractor pipelines
- `retrieval/` — retrievers, graph retriever, reranker
- `modal/` — model-serving wrappers (vision, vLLM)
- `eval/`, `fine_tuning/` — training and evaluation tooling
- `docs/` — design docs, architecture notes, and deployment guides
- `k8s/`, `terraform/` — deployment manifests and infrastructure code

Quickstart (development)
Prerequisites: Docker, Docker Compose, Python 3.10+, GNU Make

1) Run locally with Docker Compose

```bash
# start services declared in `docker-compose.yml`
docker-compose up --build
```

2) Run the API server (development)

```bash
# create virtualenv, install deps (for local development)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api/main.py
```

Deployment
- Use the manifests in `k8s/` for Kubernetes-based deployment and the Terraform modules in `terraform/` to provision cloud resources. See `docs/05_DEPLOYMENT_GUIDE.md` for detailed steps.

Contributing
- Read `docs/06_ROADMAP.md` and open an issue for any non-trivial change.
- Follow the repo's Python formatting and testing conventions; add tests for new features.

Next steps
- Configure secrets and vector store backends for your environment (see `api/config.py`).
- Try the sample ingestion pipeline in `ingestion/pipeline.py` to index a small document set.

License
- MIT (unless your organization requires a different license — update as needed)

For detailed design and rationale, see the `docs/` directory.
