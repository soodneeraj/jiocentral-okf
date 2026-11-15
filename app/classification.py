"""Document-level and page-level content classification (types A–D / page kinds)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.utils import load_config


class DocumentType(str, Enum):
    TEXT_ONLY = "A"
    TEXT_IMAGE = "B"
    IMAGE_ONLY = "C"
    MIXED = "D"
    UNKNOWN = "unknown"


class PageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text_image"
    SCANNED = "scanned"
    TABLE = "table"
    DIAGRAM = "diagram"
    EMPTY = "empty"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class PageSignals:
    page_number: int
    selectable_text_len: int = 0
    embedded_image_count: int = 0
    table_count: int = 0
    is_mostly_raster: bool = False
    has_diagram_keywords: bool = False


def classify_page(signals: PageSignals) -> PageType:
    cfg = load_config()
    text_thresh = int(cfg.get("classify_empty_text_threshold", 40))
    has_text = signals.selectable_text_len >= text_thresh
    has_images = signals.embedded_image_count > 0 or signals.is_mostly_raster
    has_tables = signals.table_count > 0

    if signals.is_mostly_raster and not has_text:
        return PageType.SCANNED
    if has_tables and not has_images and has_text:
        return PageType.TABLE
    if signals.has_diagram_keywords and has_images:
        return PageType.DIAGRAM
    if has_text and has_images:
        return PageType.TEXT_IMAGE
    if has_text and not has_images:
        return PageType.TEXT
    if has_images and not has_text:
        return PageType.IMAGE
    if not has_text and not has_images and not has_tables:
        return PageType.EMPTY
    return PageType.MIXED


def classify_document(page_types: list[PageType]) -> DocumentType:
    if not page_types:
        return DocumentType.UNKNOWN
    unique = {p for p in page_types if p != PageType.EMPTY}
    if not unique:
        return DocumentType.UNKNOWN
    if unique <= {PageType.TEXT, PageType.TABLE}:
        return DocumentType.TEXT_ONLY
    if unique <= {PageType.IMAGE, PageType.SCANNED, PageType.DIAGRAM}:
        return DocumentType.IMAGE_ONLY
    if len(unique) == 1 and PageType.TEXT_IMAGE in unique:
        return DocumentType.TEXT_IMAGE
    if unique == {PageType.TEXT, PageType.TEXT_IMAGE} or unique == {
        PageType.TEXT_IMAGE
    }:
        return DocumentType.TEXT_IMAGE
    # Multiple distinct content kinds across pages
    if len(unique) >= 2:
        return DocumentType.MIXED
    return DocumentType.TEXT_IMAGE if PageType.TEXT_IMAGE in unique else DocumentType.MIXED


def page_signals_from_pymupdf(page: Any, page_number: int) -> PageSignals:
    """Build classification signals from a PyMuPDF page object."""
    text = (page.get_text("text") or "").strip()
    images = page.get_images(full=True) or []
    # Heuristic: large full-page image xref suggests scanned/raster page
    page_area = float(page.rect.width * page.rect.height) or 1.0
    mostly_raster = False
    try:
        for img in images:
            rects = page.get_image_rects(img[0])
            for r in rects:
                if (r.width * r.height) / page_area > 0.6:
                    mostly_raster = True
                    break
            if mostly_raster:
                break
    except Exception:  # noqa: BLE001
        mostly_raster = len(images) > 0 and len(text) < 40

    lower = text.lower()
    diagram_kw = any(
        k in lower
        for k in ("architecture", "topology", "flowchart", "diagram", "network")
    )
    # Tables via text heuristics (docling supplies real counts when available)
    table_hint = text.count("|") > 8 or "\t" in text

    return PageSignals(
        page_number=page_number,
        selectable_text_len=len(text),
        embedded_image_count=len(images),
        table_count=1 if table_hint else 0,
        is_mostly_raster=mostly_raster and len(text) < 80,
        has_diagram_keywords=diagram_kw,
    )
