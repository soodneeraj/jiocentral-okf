# Multimodal RAG Platform — Architecture

## Approach

Evolve the existing Open OKF Enterprise RAG stack (FastAPI + Docling/PyMuPDF + Milvus + Ollama) into a page-aware multimodal platform. OKF Markdown remains an auditable intermediate/export; SQL + Milvus hold authoritative relationships and vectors.

## Principles

1. Every page of every PDF is processed and recorded in a page manifest.
2. Duplicate images: store binary once; retain every document/page occurrence.
3. Header/footer content is excluded from indexing, not from traceability.
4. Text and image retrieval run independently, then fuse (RRF); never mix incompatible embedding spaces.
5. Answers are grounded with document name, page, and original images when relevant.
6. All inference stays on-premise (Ollama / local OCR). No external AI APIs.

## Component map

| Component | Module |
|-----------|--------|
| Upload / jobs | `api.py`, `app/admin.py`, `app/services/worker.py` |
| Classification | `app/classification.py` |
| Page manifests | `app/manifest.py` |
| PDF ingest | `app/ingest.py` |
| OCR | `app/ocr.py` |
| Asset registry | `app/assets.py` |
| Chunking | `app/chunking.py` |
| Embeddings / Milvus | `app/index.py` |
| Hybrid retrieval | `app/retrieval.py` |
| RAG orchestration | `app/orchestrator.py` |
| Model registry | `app/models_registry.py` |
| Security / ACL | `app/security.py` |
| Conversations | SQL models + `/api/v1/chat/*` |
| React UI | `frontend/` |

## Data stores

- **Filesystem:** source PDFs, OKF bundle, image binaries
- **SQLite/Postgres (SQLAlchemy):** documents, page_manifests, image_assets, image_occurrences, chunks, conversations
- **Milvus:** `jcrag_text_chunks`, `jcrag_image_chunks` (vectors + metadata refs only)

## Ingestion flow

Validate → checksum → classify document → per-page classify/extract/OCR/render → asset registry → chunk+associate → embed → Milvus upsert → OKF export → completeness report.

## Query flow

Auth → conversation context → intent → hybrid retrieve → cross-modal associate → RRF/rerank → context assembly (+ VLM on selected images) → grounded answer + citations + images.

## Acceptance scenarios

A–R as defined in the master prompt. A phase is complete only when its scenarios pass under `eval/`.
