"""Versioned REST API: /api/v1/documents|chat|evidence|images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.admin import save_uploaded_pdfs
from app.chat import rag_answer
from app.db.models import Conversation, Document, ImageAsset, ImageOccurrence, Message
from app.db.session import get_session, init_db
from app.manifest import document_status_payload
from app.models_registry import get_model_registry
from app.security import get_current_user, issue_token, verify_password
from app.services.metrics import snapshot as metrics_snapshot
from app.services.worker import get_worker
from app.utils import okf_bundle_dir

router = APIRouter(prefix="/api/v1")


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatQueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    top_k: int = 10
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str | None = None


class TransformRequest(BaseModel):
    action: str  # regenerate | shorter | detailed | simpler
    query: str
    answer: str = ""
    excerpts: str = ""
    conversation_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int = 10


class FeedbackRequest(BaseModel):
    rating: str  # up | down
    conversation_id: str | None = None
    message_id: str | None = None
    query: str = ""
    note: str = ""


class ProcessBody(BaseModel):
    force: bool = False
    do_index: bool = True
    recreate: bool = False


@router.post("/auth/login")
def login(body: LoginRequest) -> dict[str, Any]:
    user = verify_password(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = issue_token(user["username"], user["roles"], user["access_scopes"])
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.get("/health")
def health() -> dict[str, Any]:
    from app.index import get_milvus_client
    from app.utils import load_config

    cfg = load_config()
    milvus = "unknown"
    try:
        client = get_milvus_client()
        milvus = (
            "ok"
            if client.has_collection(cfg["text_collection"])
            and client.has_collection(cfg["image_collection"])
            else "collection_missing"
        )
    except Exception as exc:  # noqa: BLE001
        milvus = f"error: {exc}"
    return {
        "milvus": milvus,
        "models": get_model_registry().health(),
        "metrics": metrics_snapshot(),
    }


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    return metrics_snapshot()


# ----------------------------- documents ---------------------------------


@router.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    process: bool = True,
    force: bool = False,
    do_index: bool = True,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        name = f.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            continue
        data = await f.read()
        if data:
            payloads.append((name, data))
    if not payloads:
        raise HTTPException(status_code=400, detail="No PDF files received")
    saved = save_uploaded_pdfs(payloads)
    job = None
    if process:
        job_obj = get_worker().enqueue(
            "pipeline",
            {"force": force, "do_index": do_index, "recreate": False, "files": saved},
        )
        job = {"id": job_obj.id, "state": job_obj.state}
    return {"saved": saved, "job": job, "uploaded_by": user.get("username")}


@router.get("/documents")
def list_documents(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    scopes = user.get("access_scopes") or []
    with get_session() as session:
        docs = session.query(Document).order_by(Document.upload_timestamp.desc()).all()
        rows = []
        for d in docs:
            if scopes and d.access_scope not in scopes and "admin" not in (user.get("roles") or []):
                continue
            rows.append(
                {
                    "document_id": d.id,
                    "name": d.name,
                    "original_filename": d.original_filename,
                    "page_count": d.page_count,
                    "document_type": d.document_type,
                    "processing_status": d.processing_status,
                    "file_checksum": d.file_checksum,
                    "access_scope": d.access_scope,
                    "upload_timestamp": d.upload_timestamp.isoformat() if d.upload_timestamp else None,
                }
            )
    return {"documents": rows}


@router.get("/documents/{document_id}")
def get_document(document_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        d = session.get(Document, document_id)
        if not d:
            raise HTTPException(status_code=404, detail="Document not found")
        return document_status_payload(session, d)


@router.get("/documents/{document_id}/status")
def document_status(document_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return get_document(document_id, user)


@router.get("/documents/{document_id}/pages")
def document_pages(document_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    payload = get_document(document_id, user)
    return {"document_id": document_id, "pages": payload.get("pages") or []}


@router.get("/documents/{document_id}/processing-report")
def processing_report(document_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    payload = get_document(document_id, user)
    return {"document_id": document_id, "report": payload.get("report")}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if "admin" not in (user.get("roles") or []) and __import__(
        "app.security", fromlist=["auth_enabled"]
    ).auth_enabled():
        raise HTTPException(status_code=403, detail="Admin required")
    init_db()
    with get_session() as session:
        d = session.get(Document, document_id)
        if not d:
            raise HTTPException(status_code=404, detail="Document not found")
        filename = d.original_filename
    from app.admin import purge_source_pdf

    result = purge_source_pdf(filename)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("detail") or "Delete failed")
    return {"deleted": document_id, "purge": result}


@router.post("/documents/process")
def process_documents(
    body: ProcessBody,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    job = get_worker().enqueue(
        "pipeline",
        {"force": body.force, "do_index": body.do_index, "recreate": body.recreate},
    )
    return {"job": {"id": job.id, "state": job.state}}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_worker().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "state": job.state,
        "message": job.message,
        "result": job.result,
        "finished_at": job.finished_at,
    }


# ----------------------------- chat --------------------------------------


@router.post("/chat/query")
def chat_query(
    body: ChatQueryRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    from app.services.metrics import Timer, incr

    incr("chat.queries")
    with Timer("chat.latency"):
        result = rag_answer(
            body.query,
            top_k=body.top_k,
            history=body.history or None,
            conversation_id=body.conversation_id,
            user_id=str(user.get("username") or "anonymous"),
            access_scopes=list(user.get("access_scopes") or []),
            mode=body.mode,
        )
    return result


@router.post("/chat/transform")
def chat_transform(
    body: TransformRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    from app.answer_actions import transform_answer

    action = (body.action or "").strip().lower()
    if action not in {"regenerate", "shorter", "detailed", "simpler"}:
        raise HTTPException(status_code=400, detail="Invalid action")
    return transform_answer(
        action=action,  # type: ignore[arg-type]
        query=body.query,
        answer=body.answer,
        excerpts=body.excerpts,
        history=body.history or None,
        conversation_id=body.conversation_id,
        user_id=str(user.get("username") or "anonymous"),
        top_k=body.top_k,
    )


@router.post("/chat/feedback")
def chat_feedback(
    body: FeedbackRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    from app.answer_actions import save_feedback

    rating = (body.rating or "").strip().lower()
    if rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be up or down")
    return save_feedback(
        rating=rating,  # type: ignore[arg-type]
        conversation_id=body.conversation_id,
        message_id=body.message_id,
        query=body.query,
        note=body.note,
        user_id=str(user.get("username") or "anonymous"),
    )


@router.get("/chat/conversations")
def list_conversations(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        rows = (
            session.query(Conversation)
            .filter_by(user_id=str(user.get("username") or "anonymous"))
            .order_by(Conversation.updated_at.desc())
            .limit(100)
            .all()
        )
        return {
            "conversations": [
                {
                    "conversation_id": c.id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in rows
            ]
        }


@router.get("/chat/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        c = session.get(Conversation, conversation_id)
        if not c:
            raise HTTPException(status_code=404, detail="Conversation not found")
        msgs = (
            session.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return {
            "conversation_id": c.id,
            "title": c.title,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "evidence": m.evidence,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in msgs
            ],
        }


@router.delete("/chat/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        c = session.get(Conversation, conversation_id)
        if not c:
            raise HTTPException(status_code=404, detail="Conversation not found")
        session.delete(c)
    return {"deleted": conversation_id}


# ----------------------------- evidence / images -------------------------


@router.get("/evidence/{evidence_id}")
def get_evidence(evidence_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        from app.db.models import ChunkRecord

        chunk = session.get(ChunkRecord, evidence_id)
        if chunk:
            return {
                "evidence_id": chunk.id,
                "type": "chunk",
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "section_heading": chunk.section_heading,
                "text": chunk.chunk_text,
                "associated_image_ids": chunk.associated_image_ids,
            }
        occ = session.get(ImageOccurrence, evidence_id)
        if occ:
            asset = session.get(ImageAsset, occ.asset_id)
            return {
                "evidence_id": occ.id,
                "type": "image_occurrence",
                "document_id": occ.document_id,
                "page_number": occ.page_number,
                "image_path": asset.image_path if asset else None,
                "caption": occ.caption,
            }
    raise HTTPException(status_code=404, detail="Evidence not found")


@router.get("/documents/{document_id}/pages/{page_number}")
def get_page(
    document_id: str,
    page_number: int,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    payload = get_document(document_id, user)
    for p in payload.get("pages") or []:
        if int(p.get("page_number") or 0) == page_number:
            return p
    raise HTTPException(status_code=404, detail="Page not found")


@router.get("/images/{image_id}")
def get_image(image_id: str, user: dict[str, Any] = Depends(get_current_user)) -> Any:
    init_db()
    with get_session() as session:
        occ = session.get(ImageOccurrence, image_id)
        asset = None
        if occ:
            asset = session.get(ImageAsset, occ.asset_id)
        if asset is None:
            asset = session.get(ImageAsset, image_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Image not found")
        # ACL: occurrence inherits document access
        if occ:
            doc = session.get(Document, occ.document_id)
            scopes = user.get("access_scopes") or []
            if (
                doc
                and scopes
                and doc.access_scope not in scopes
                and "admin" not in (user.get("roles") or [])
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
        path = okf_bundle_dir() / asset.image_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image file missing")
        return FileResponse(path)


@router.get("/images/assets/{asset_id}")
def get_asset(asset_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        asset = session.get(ImageAsset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        return {
            "asset_id": asset.id,
            "content_hash": asset.content_hash,
            "perceptual_hash": asset.perceptual_hash,
            "image_path": asset.image_path,
            "width": asset.width,
            "height": asset.height,
            "image_description": asset.image_description,
            "ocr_text": asset.ocr_text,
        }


@router.get("/images/{image_id}/occurrences")
def image_occurrences(image_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    init_db()
    with get_session() as session:
        asset = session.get(ImageAsset, image_id)
        if not asset:
            occ = session.get(ImageOccurrence, image_id)
            if not occ:
                raise HTTPException(status_code=404, detail="Not found")
            asset = session.get(ImageAsset, occ.asset_id)
        assert asset is not None
        scopes = user.get("access_scopes") or []
        rows = []
        for occ in asset.occurrences:
            doc = session.get(Document, occ.document_id)
            if (
                doc
                and scopes
                and doc.access_scope not in scopes
                and "admin" not in (user.get("roles") or [])
            ):
                continue
            rows.append(
                {
                    "occurrence_id": occ.id,
                    "document_id": occ.document_id,
                    "page_number": occ.page_number,
                    "image_role": occ.image_role,
                    "indexable": occ.indexable,
                }
            )
        return {"asset_id": asset.id, "occurrences": rows}
