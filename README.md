# Open OKF Enterprise RAG

Page-aware, multimodal, fully on-premise document intelligence platform: upload PDFs → classify every page → extract text/images/tables → OCR scanned pages → dedupe image binaries while retaining occurrences → chunk + embed into Milvus → hybrid text/image retrieval → grounded answers with citations and original figures.

Intermediate OKF Markdown remains an audit/export format. Authoritative metadata lives in SQL (`data/okf_metadata.db`). No cloud AI APIs.

See [docs/architecture.md](docs/architecture.md) for the full blueprint.

## Architecture

```
source_pdfs/*.pdf
      │  page-aware pipeline (app/pipeline.py)
      ▼
SQL manifests + image asset registry + okf-bundle/
      │
      ▼
app/index.py  →  Milvus (text + image collections)
      │
      ▼
hybrid retrieval + orchestrator → api.py (:8000)
      ├─ /api/v1/* REST
      ├─ /app/ React UI (frontend/dist)
      └─ ui.py Gradio (legacy :7860)
```

Milvus is **not** run by this project — connect to an existing server.

## Requirements

- Python 3.11
- [Ollama](https://ollama.com) reachable at `ollama_url`
- [Milvus](https://milvus.io) reachable at `milvus_uri`
- Models pulled into Ollama:
  - Embedding: `qllama/bge-large-en-v1.5:f16` (1024-dim)
  - Chat: `mistral:7b` (default), fallback `llama3.1:8b`
  - Vision: `llava:13b`

## Quick start

```bash
# 1. Install
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your MILVUS_URI / OLLAMA_URL / credentials

# 3. Pull Ollama models
ollama pull qllama/bge-large-en-v1.5:f16
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull llava:13b

# 4. Drop PDFs into source_pdfs/, then:
python -m app.ingest
python -m app.index

# 5. Serve
uvicorn api:app --host 0.0.0.0 --port 8000
# In another terminal:
python ui.py
```

API: http://localhost:8000  
UI:  http://localhost:7860  
Health: `GET /health`

## Docker

```bash
docker-compose up -d
# Pull models into the ollama service:
docker-compose exec ollama ollama pull qllama/bge-large-en-v1.5:f16
docker-compose exec ollama ollama pull mistral:7b
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull llava:13b
```

The container default CMD starts only FastAPI (`:8000`). Run the Gradio UI separately (`python ui.py`) if needed. Inside Compose, `OLLAMA_URL` is `http://ollama:11434` (service DNS), not `localhost`.

## CLI reference

| Command | Purpose |
|---|---|
| `python -m app.ingest` | Ingest all PDFs under `source_pdfs/` (skips already-ingested docs) |
| `python -m app.ingest --force` | Destructive re-ingest (clears concepts **after** successful convert) |
| `python -m app.index` | Embed OKF concepts into Milvus; captions figures once |
| `python -m app.index --recreate` | Drop & recreate collections, then reindex |
| `python -m app.exclude_image "assets/images/<doc>_p3_fig0.png"` | Mark a figure excluded (takes effect after `--recreate`) |

## HTTP API

Legacy routes remain. Prefer `/api/v1/*`.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Milvus + Ollama + model registry |
| `POST` | `/chat` | Legacy RAG answer |
| `POST` | `/api/v1/documents/upload` | Multi-PDF upload + queued pipeline |
| `GET` | `/api/v1/documents` | List documents + status |
| `GET` | `/api/v1/documents/{id}/processing-report` | Page completeness report |
| `POST` | `/api/v1/chat/query` | Hybrid multimodal RAG |
| `GET` | `/api/v1/images/{id}` | Secure image delivery |
| `GET` | `/app/` | Chat UI |

## Evaluation

```bash
python -m eval.scenarios
# Optional live stack:
# OKF_LIVE_EVAL=1 python -m eval.scenarios
```

## Config

`config.yaml` is the single source of truth. Any key can be overridden by an uppercase env var (e.g. `MILVUS_URI`). Load `.env` via `python-dotenv` at import time. Never commit real endpoints/credentials — use `.env` (gitignored).

Optional brand/logo filtering: drop reference images into `app/reference_images/` (near-blank references with grayscale std &lt; 3.0 are refused at load time).

## Known limitations

1. Tables/figures have empty `section` (extraction loops are not interleaved with the text heading trail).
2. No per-concept Milvus update/delete — figure exclusion or frontmatter changes require `--recreate`.
3. `ambiguity_score_gap_threshold: 0.03` is an unvalidated starting guess.
4. `/ingest` and `/index` have no job-status API.
5. Older concept files lack `version`/`section`/`excluded` until a `--force` ingest + `--recreate` index.

## License

Open source — use freely. No cloud vendor lock-in.
