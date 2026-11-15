"""Simple in-process ingest/index queue so chat is not blocked by long jobs."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    state: str = "queued"  # queued|running|done|error
    message: str = ""
    created_at: str = ""
    finished_at: str | None = None
    result: dict[str, Any] | None = None


class IngestWorker:
    def __init__(self) -> None:
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, Callable[[Job], dict[str, Any]]] = {}

    def register(self, kind: str, handler: Callable[[Job], dict[str, Any]]) -> None:
        self._handlers[kind] = handler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="okf-worker")
        self._thread.start()
        logger.info("Ingest worker started")

    def enqueue(self, kind: str, payload: dict[str, Any] | None = None) -> Job:
        job = Job(
            id=str(uuid4()),
            kind=kind,
            payload=payload or {},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        self.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
            return [self._as_dict(j) for j in jobs]

    def _as_dict(self, j: Job) -> dict[str, Any]:
        return {
            "id": j.id,
            "kind": j.kind,
            "state": j.state,
            "message": j.message,
            "created_at": j.created_at,
            "finished_at": j.finished_at,
            "result": j.result,
            "payload": j.payload,
        }

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            handler = self._handlers.get(job.kind)
            job.state = "running"
            job.message = f"Running {job.kind}"
            try:
                if not handler:
                    raise RuntimeError(f"No handler for job kind {job.kind}")
                job.result = handler(job)
                job.state = "done"
                job.message = "Completed"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s failed", job.id)
                job.state = "error"
                job.message = str(exc)
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                self._q.task_done()


_worker: IngestWorker | None = None


def get_worker() -> IngestWorker:
    global _worker
    if _worker is None:
        _worker = IngestWorker()

        def _ingest(job: Job) -> dict[str, Any]:
            from app.pipeline import PageAwareIngestor

            return PageAwareIngestor().ingest_all(force=bool(job.payload.get("force")))

        def _index(job: Job) -> dict[str, Any]:
            from app.index import index_okf

            return index_okf(recreate=bool(job.payload.get("recreate")))

        def _pipeline(job: Job) -> dict[str, Any]:
            from app.index import index_okf
            from app.pipeline import PageAwareIngestor

            force = bool(job.payload.get("force"))
            ingest = PageAwareIngestor().ingest_all(force=force)
            index_stats = {}
            if job.payload.get("do_index", True):
                index_stats = index_okf(recreate=bool(job.payload.get("recreate")))
            return {"ingest": ingest, "index": index_stats}

        _worker.register("ingest", _ingest)
        _worker.register("index", _index)
        _worker.register("pipeline", _pipeline)
        if __import__("app.utils", fromlist=["load_config"]).load_config().get(
            "ingest_queue_enabled", True
        ):
            _worker.start()
    return _worker


def main() -> None:
    """Entrypoint for docker-compose worker profile."""
    import time

    w = get_worker()
    w.start()
    logger.info("Standalone worker process running")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
