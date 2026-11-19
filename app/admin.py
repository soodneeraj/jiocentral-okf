"""Admin helpers: save uploaded PDFs into source_pdfs/ and kick off RAG processing."""

from __future__ import annotations

import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from app.index import delete_by_document_id, delete_by_source, index_okf
from app.logger import get_logger
from app.utils import (
    concepts_dir,
    images_dir,
    okf_bundle_dir,
    source_pdfs_dir,
    tables_dir,
)

logger = get_logger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ()\[\]]+")

_job_lock = threading.Lock()
_job_status: dict[str, Any] = {
    "state": "idle",  # idle | running | done | error
    "message": "No admin job running.",
    "started_at": None,
    "finished_at": None,
    "files": [],
}


def get_job_status() -> dict[str, Any]:
    with _job_lock:
        return dict(_job_status)


def list_source_pdfs() -> list[dict[str, Any]]:
    src = source_pdfs_dir()
    src.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(src.glob("*.pdf")):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return rows


def _unlink_glob(root: Path, pattern: str) -> list[str]:
    removed: list[str] = []
    if not root.exists():
        return removed
    for path in root.glob(pattern):
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
                removed.append(str(path.relative_to(okf_bundle_dir())))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove %s: %s", path, exc)
    return removed


def _purge_orphaned_assets(session: Any, asset_ids: set[str]) -> dict[str, Any]:
    """Delete ImageAsset rows (and registry files) that no longer have occurrences."""
    from app.db.models import ImageAsset, ImageOccurrence

    removed_assets = 0
    removed_files: list[str] = []
    bundle = okf_bundle_dir()
    for aid in asset_ids:
        if not aid:
            continue
        remaining = (
            session.query(ImageOccurrence).filter(ImageOccurrence.asset_id == aid).count()
        )
        if remaining:
            continue
        asset = session.get(ImageAsset, aid)
        if not asset:
            continue
        rel = str(asset.image_path or "").strip()
        session.delete(asset)
        removed_assets += 1
        if rel:
            abs_path = bundle / rel
            try:
                if abs_path.is_file():
                    abs_path.unlink(missing_ok=True)
                    removed_files.append(rel)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove registry asset %s: %s", abs_path, exc)
    return {"assets": removed_assets, "registry_files": removed_files}


def purge_source_pdf(filename: str) -> dict[str, Any]:
    """
    Remove a source PDF and every persistent reference:
    source_pdfs file, OKF concepts/images/tables, Milvus rows, SQLite metadata,
    and orphaned image registry assets.
    """
    with _job_lock:
        if _job_status.get("state") == "running":
            return {
                "ok": False,
                "detail": "Cannot delete while an admin ingest/index job is running.",
            }

    name = Path(str(filename or "")).name
    if not name or not name.lower().endswith(".pdf"):
        return {"ok": False, "detail": "Invalid PDF filename."}
    if ".." in name or "/" in str(filename) or "\\" in str(filename):
        # Path(name).name already strips dirs; keep an explicit guard for odd inputs
        if name != Path(str(filename)).name:
            return {"ok": False, "detail": "Invalid PDF filename."}

    stem = Path(name).stem
    src_path = source_pdfs_dir() / name
    report: dict[str, Any] = {
        "ok": True,
        "filename": name,
        "stem": stem,
        "pdf_removed": False,
        "concepts_removed": False,
        "images_removed": [],
        "tables_removed": [],
        "documents": [],
        "milvus": {"text": 0, "image": 0},
        "db": {"documents": 0, "orphaned_assets": {}},
    }

    # --- SQLite metadata ---
    try:
        from app.db.models import Document, ImageOccurrence
        from app.db.session import get_session, init_db

        init_db()
        with get_session() as session:
            docs = (
                session.query(Document)
                .filter(
                    (Document.original_filename == name)
                    | (Document.name == stem)
                    | (Document.storage_path.endswith(name))
                )
                .all()
            )
            # Prefer exact filename matches when both stem-shared rows exist
            exact = [d for d in docs if d.original_filename == name]
            if exact:
                docs = exact

            asset_ids: set[str] = set()
            for d in docs:
                for occ in (
                    session.query(ImageOccurrence)
                    .filter(ImageOccurrence.document_id == d.id)
                    .all()
                ):
                    if occ.asset_id:
                        asset_ids.add(str(occ.asset_id))
                report["documents"].append(
                    {"id": d.id, "name": d.name, "original_filename": d.original_filename}
                )
                session.delete(d)
                report["db"]["documents"] += 1

            session.flush()
            report["db"]["orphaned_assets"] = _purge_orphaned_assets(session, asset_ids)
            session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("DB purge failed for %s", name)
        report["db"]["error"] = str(exc)

    # --- Milvus ---
    milvus_text = 0
    milvus_image = 0
    for doc in report["documents"]:
        try:
            counts = delete_by_document_id(str(doc["id"]))
            milvus_text += int(counts.get("text") or 0)
            milvus_image += int(counts.get("image") or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus delete_by_document_id failed: %s", exc)
    try:
        by_source = delete_by_source(name)
        milvus_text += int(by_source.get("text") or 0)
        milvus_image += int(by_source.get("image") or 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus delete_by_source failed: %s", exc)
    report["milvus"] = {"text": milvus_text, "image": milvus_image}

    # --- OKF filesystem ---
    concept_dir = concepts_dir() / stem
    if concept_dir.exists() and concept_dir.is_dir():
        shutil.rmtree(concept_dir, ignore_errors=True)
        report["concepts_removed"] = not concept_dir.exists()

    report["images_removed"] = _unlink_glob(images_dir(), f"{stem}_*")
    report["tables_removed"] = _unlink_glob(tables_dir(), f"{stem}_*")

    if src_path.exists():
        try:
            src_path.unlink()
            report["pdf_removed"] = True
        except Exception as exc:  # noqa: BLE001
            report["ok"] = False
            report["detail"] = f"Failed to delete PDF file: {exc}"
            logger.warning("Failed to delete %s: %s", src_path, exc)
    else:
        report["pdf_removed"] = False
        if not report["documents"] and not report["concepts_removed"]:
            report["ok"] = False
            report["detail"] = f"PDF not found: {name}"

    # --- Best-effort ES cache purge for answers citing this source ---
    try:
        from app.cache_es import get_es_client
        from app.utils import load_config

        cfg = load_config()
        if cfg.get("elasticsearch_cache_enabled", True):
            client = get_es_client()
            if client is not None:
                index = cfg.get("elasticsearch_cache_index") or "okf_qa_cache"
                if client.indices.exists(index=index):
                    resp = client.delete_by_query(
                        index=index,
                        body={
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match_phrase": {"answer": name}},
                                        {"match_phrase": {"sources.source": name}},
                                    ],
                                    "minimum_should_match": 1,
                                }
                            }
                        },
                        refresh=True,
                        conflicts="proceed",
                    )
                    report["es_cache_deleted"] = int(resp.get("deleted") or 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ES cache purge skipped: %s", exc)
        report["es_cache_deleted"] = 0

    logger.info("Purged source PDF %s → %s", name, report)
    return report


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = _SAFE_NAME_RE.sub("_", base).strip(" ._")
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    if not cleaned or cleaned == ".pdf":
        cleaned = f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    return cleaned


def save_uploaded_pdfs(
    files: list[tuple[str, BinaryIO | bytes | Path]],
) -> list[str]:
    """Save uploaded PDF bytes/paths into source_pdfs/. Returns saved filenames."""
    src = source_pdfs_dir()
    src.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for original_name, payload in files:
        filename = _safe_filename(original_name)
        dest = src / filename
        # Avoid silent overwrite of unrelated content: uniquify if exists
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dest = src / f"{stem}_{stamp}{suffix}"
            filename = dest.name

        if isinstance(payload, Path):
            shutil.copy2(payload, dest)
        elif isinstance(payload, (bytes, bytearray)):
            dest.write_bytes(payload)
        else:
            # file-like
            data = payload.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            dest.write_bytes(data)

        saved.append(filename)
        logger.info("Admin uploaded PDF → %s", dest)
    return saved


def _run_pipeline(force: bool, do_index: bool, recreate: bool, files: list[str]) -> None:
    global _job_status
    try:
        with _job_lock:
            _job_status.update(
                {
                    "state": "running",
                    "message": f"Ingesting {len(files)} file(s)…",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": None,
                    "files": files,
                }
            )
        from app.pipeline import PageAwareIngestor

        ingest_result = PageAwareIngestor().ingest_all(force=force)
        n = ingest_result.get("processed", 0)
        msg = f"Ingest complete ({n} PDF(s) processed)."
        if do_index:
            with _job_lock:
                _job_status["message"] = msg + " Indexing into Milvus…"
            stats = index_okf(recreate=recreate)
            msg = (
                f"{msg} Index complete "
                f"(text={stats.get('text', 0)}, image={stats.get('image', 0)}, "
                f"errors={stats.get('errors', 0)})."
            )
        with _job_lock:
            _job_status.update(
                {
                    "state": "done",
                    "message": msg,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        logger.info("Admin pipeline finished: %s", msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Admin pipeline failed")
        with _job_lock:
            _job_status.update(
                {
                    "state": "error",
                    "message": f"Pipeline failed: {exc}",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )


def start_pipeline(
    *,
    force: bool = False,
    do_index: bool = True,
    recreate: bool = False,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Start ingest (+ optional index) in a background thread. Rejects if busy."""
    with _job_lock:
        if _job_status.get("state") == "running":
            return {
                "accepted": False,
                "status": dict(_job_status),
                "detail": "Another admin job is already running.",
            }
        _job_status.update(
            {
                "state": "running",
                "message": "Starting…",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "files": list(files or []),
            }
        )

    thread = threading.Thread(
        target=_run_pipeline,
        kwargs={
            "force": force,
            "do_index": do_index,
            "recreate": recreate,
            "files": list(files or []),
        },
        daemon=True,
        name="okf-admin-pipeline",
    )
    thread.start()
    return {"accepted": True, "status": get_job_status()}
