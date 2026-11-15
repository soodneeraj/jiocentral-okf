"""FastAPI entrypoint for Open OKF Enterprise RAG."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.admin import (
    get_job_status,
    list_source_pdfs,
    save_uploaded_pdfs,
    start_pipeline,
)
from app.api_v1 import router as api_v1_router
from app.cache_es import elasticsearch_health
from app.chat import rag_answer
from app.db.session import init_db
from app.index import get_milvus_client
from app.logger import get_logger
from app.models_registry import get_model_registry
from app.services.worker import get_worker
from app.utils import PROJECT_ROOT, images_dir, load_config, okf_bundle_dir

logger = get_logger(__name__)

app = FastAPI(title="Open OKF Enterprise RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_v1_router)


@app.on_event("startup")
def _on_startup() -> None:
    init_db()
    get_worker()
    logger.info("API startup complete (DB + worker)")

_ADMIN_PAGE = (PROJECT_ROOT / "app" / "admin_page.html").read_text(encoding="utf-8")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    top_k: int = 10
    history: list[HistoryMessage] = Field(default_factory=list)
    conversation_id: str | None = None


class ProcessRequest(BaseModel):
    force: bool = False
    do_index: bool = True
    recreate: bool = False


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Open OKF Enterprise RAG",
        "description": (
            "Page-aware multimodal RAG: PDFs → OKF + SQL manifests → Milvus → "
            "hybrid retrieval → local Ollama. No cloud AI APIs."
        ),
        "endpoints": {
            "legacy": "/health, /chat, /ingest, /index, /admin/*",
            "v1": "/api/v1/documents/*, /api/v1/chat/*, /api/v1/evidence/*, /api/v1/images/*",
        },
        "ui": "/app/",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Probe Milvus and Ollama independently; never raises."""
    cfg = load_config()
    result: dict[str, Any] = {
        "milvus": "unknown",
        "ollama": "unknown",
        "elasticsearch": "unknown",
        "chat_model": cfg.get("chat_model"),
        "embedding_model": cfg.get("embedding_model"),
    }

    # Milvus
    try:
        client = get_milvus_client()
        text_ok = client.has_collection(cfg["text_collection"])
        image_ok = client.has_collection(cfg["image_collection"])
        if text_ok and image_ok:
            result["milvus"] = "ok"
        else:
            result["milvus"] = (
                f"collection_missing (text={text_ok}, images={image_ok})"
            )
    except Exception as exc:  # noqa: BLE001
        result["milvus"] = f"error: {exc}"

    # Ollama
    try:
        import ollama

        oclient = ollama.Client(host=cfg["ollama_url"])
        listing = oclient.list()
        if isinstance(listing, dict):
            models_raw = listing.get("models") or []
        else:
            models_raw = getattr(listing, "models", None) or []
        names: list[str] = []
        for m in models_raw:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model") or ""
            else:
                name = getattr(m, "name", None) or getattr(m, "model", "") or ""
            if name:
                names.append(str(name))
        result["ollama"] = "ok"
        result["ollama_models"] = names
    except Exception as exc:  # noqa: BLE001
        result["ollama"] = f"error: {exc}"

    # Elasticsearch Q&A cache
    try:
        result["elasticsearch"] = elasticsearch_health()
    except Exception as exc:  # noqa: BLE001
        result["elasticsearch"] = {"status": f"error: {exc}"}

    try:
        result["models"] = get_model_registry().health()
    except Exception as exc:  # noqa: BLE001
        result["models"] = {"error": str(exc)}

    return result


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    history = [{"role": m.role, "content": m.content} for m in req.history] or None
    return rag_answer(
        req.query,
        top_k=req.top_k,
        history=history,
        conversation_id=req.conversation_id,
    )


# Serve React build if present; also expose image assets
_frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/app", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")

_images = images_dir()
_images.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/images",
    StaticFiles(directory=str(_images)),
    name="images",
)
app.mount(
    "/static/bundle",
    StaticFiles(directory=str(okf_bundle_dir())),
    name="bundle",
)


@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks) -> dict[str, Any]:
    job = get_worker().enqueue("ingest", {"force": False})
    return {"status": "ingestion queued", "job_id": job.id}


@app.post("/index")
def index(background_tasks: BackgroundTasks) -> dict[str, Any]:
    job = get_worker().enqueue("index", {"recreate": False})
    return {"status": "indexing queued", "job_id": job.id}


# ---------------------------------------------------------------------------
# Admin — PDF upload into RAG backend
# ---------------------------------------------------------------------------


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> str:
    """Browser admin console for uploading PDFs into the RAG pipeline."""
    return _ADMIN_PAGE


@app.get("/admin/pdfs")
def admin_list_pdfs() -> dict[str, Any]:
    return {"pdfs": list_source_pdfs()}


@app.get("/admin/status")
def admin_status() -> dict[str, Any]:
    return get_job_status()


@app.post("/admin/upload")
async def admin_upload(
    files: list[UploadFile] = File(...),
    force: str = Form("false"),
    do_index: str = Form("true"),
    recreate: str = Form("false"),
    process: str = Form("true"),
) -> dict[str, Any]:
    """Upload one or more PDFs into source_pdfs/; optionally ingest+index."""
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        name = f.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            continue
        data = await f.read()
        if not data:
            continue
        payloads.append((name, data))

    if not payloads:
        return {"saved": [], "detail": "No PDF files received.", "job": None}

    saved = save_uploaded_pdfs(payloads)
    job = None
    if _as_bool(process):
        job = start_pipeline(
            force=_as_bool(force),
            do_index=_as_bool(do_index),
            recreate=_as_bool(recreate),
            files=saved,
        )
    return {"saved": saved, "job": job}


@app.post("/admin/process")
def admin_process(req: ProcessRequest) -> dict[str, Any]:
    """Run ingest (+ optional index) on whatever is already in source_pdfs/."""
    return start_pipeline(
        force=req.force,
        do_index=req.do_index,
        recreate=req.recreate,
        files=[p["name"] for p in list_source_pdfs()],
    )
