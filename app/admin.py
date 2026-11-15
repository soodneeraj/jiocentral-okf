"""Admin helpers: save uploaded PDFs into source_pdfs/ and kick off RAG processing."""

from __future__ import annotations

import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from app.index import index_okf
from app.logger import get_logger
from app.utils import source_pdfs_dir

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
