"""Page-level processing manifest and ingestion completeness reports."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document, IngestionReport, PageManifest
from app.logger import get_logger

logger = get_logger(__name__)

TERMINAL_OUTCOMES = frozenset(
    {
        "text_ok",
        "image_ok",
        "ocr_ok",
        "empty",
        "duplicate_referenced",
        "failed",
        "n/a",
    }
)

STAGE_FIELDS = (
    "text_extraction_status",
    "ocr_status",
    "image_extraction_status",
    "table_extraction_status",
    "diagram_analysis_status",
    "duplicate_detection_status",
    "chunking_status",
    "embedding_status",
    "indexing_status",
)


def ensure_page_rows(session: Session, document: Document, page_count: int) -> list[PageManifest]:
    """Create pending manifest rows for every page (1-indexed). Never skip a page."""
    existing = {
        p.page_number: p
        for p in session.query(PageManifest).filter_by(document_id=document.id).all()
    }
    rows: list[PageManifest] = []
    for n in range(1, page_count + 1):
        if n in existing:
            rows.append(existing[n])
            continue
        row = PageManifest(document_id=document.id, page_number=n, outcome="pending")
        session.add(row)
        rows.append(row)
    session.flush()
    document.page_count = page_count
    return rows


def get_page(session: Session, document_id: str, page_number: int) -> PageManifest | None:
    return (
        session.query(PageManifest)
        .filter_by(document_id=document_id, page_number=page_number)
        .one_or_none()
    )


def update_page(
    session: Session,
    document_id: str,
    page_number: int,
    *,
    page_type: str | None = None,
    outcome: str | None = None,
    extracted_content_count: int | None = None,
    error: str | None = None,
    **stage_updates: str,
) -> PageManifest:
    row = get_page(session, document_id, page_number)
    if row is None:
        row = PageManifest(document_id=document_id, page_number=page_number)
        session.add(row)
        session.flush()
    if page_type is not None:
        row.page_type = page_type
    if outcome is not None:
        row.outcome = outcome
    if extracted_content_count is not None:
        row.extracted_content_count = extracted_content_count
    if error:
        errs = list(row.processing_errors or [])
        errs.append(error)
        row.processing_errors = errs
    for key, value in stage_updates.items():
        if key in STAGE_FIELDS:
            setattr(row, key, value)
    session.flush()
    return row


def build_completeness_report(session: Session, document: Document) -> dict[str, Any]:
    pages = (
        session.query(PageManifest)
        .filter_by(document_id=document.id)
        .order_by(PageManifest.page_number)
        .all()
    )
    failed = [p.page_number for p in pages if p.outcome == "failed"]
    pending = [p.page_number for p in pages if p.outcome == "pending"]
    processed = [p for p in pages if p.outcome in TERMINAL_OUTCOMES]
    complete = (
        len(pages) == document.page_count
        and document.page_count > 0
        and not failed
        and not pending
        and all(p.outcome in TERMINAL_OUTCOMES for p in pages)
    )
    summary = {
        "document_id": document.id,
        "document_name": document.name,
        "pages_total": document.page_count,
        "pages_with_manifest": len(pages),
        "pages_processed": len(processed),
        "pages_failed": failed,
        "pages_pending": pending,
        "outcomes": {str(p.page_number): p.outcome for p in pages},
        "page_types": {str(p.page_number): p.page_type for p in pages},
        "complete": complete,
    }
    report = IngestionReport(
        document_id=document.id,
        complete=complete,
        pages_total=document.page_count,
        pages_processed=len(processed),
        pages_failed=failed,
        summary=summary,
    )
    session.add(report)
    if complete:
        document.processing_status = "completed"
    elif failed and processed:
        document.processing_status = "partial"
    elif failed and not processed:
        document.processing_status = "failed"
    else:
        document.processing_status = "processing"
    session.flush()
    return summary


def document_status_payload(session: Session, document: Document) -> dict[str, Any]:
    pages = (
        session.query(PageManifest)
        .filter_by(document_id=document.id)
        .order_by(PageManifest.page_number)
        .all()
    )
    latest = (
        session.query(IngestionReport)
        .filter_by(document_id=document.id)
        .order_by(IngestionReport.created_at.desc())
        .first()
    )
    return {
        "document_id": document.id,
        "name": document.name,
        "processing_status": document.processing_status,
        "document_type": document.document_type,
        "page_count": document.page_count,
        "file_checksum": document.file_checksum,
        "pages": [
            {
                "page_number": p.page_number,
                "page_type": p.page_type,
                "outcome": p.outcome,
                "text_extraction_status": p.text_extraction_status,
                "ocr_status": p.ocr_status,
                "image_extraction_status": p.image_extraction_status,
                "table_extraction_status": p.table_extraction_status,
                "chunking_status": p.chunking_status,
                "embedding_status": p.embedding_status,
                "indexing_status": p.indexing_status,
                "extracted_content_count": p.extracted_content_count,
                "processing_errors": p.processing_errors or [],
            }
            for p in pages
        ],
        "report": latest.summary if latest else None,
    }
