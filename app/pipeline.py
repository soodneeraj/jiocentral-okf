"""Page-aware multimodal ingestion pipeline (Phases 2–3)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from app.assets import register_image
from app.chunking import associate_images_to_chunks, chunk_table_markdown, link_chunks, pack_paragraphs
from app.classification import (
    DocumentType,
    PageType,
    classify_document,
    classify_page,
    page_signals_from_pymupdf,
)
from app.db.models import ChunkRecord, Document
from app.db.session import get_session, init_db
from app.logger import get_logger
from app.manifest import build_completeness_report, ensure_page_rows, update_page
from app.models_registry import get_model_registry
from app.ocr import ocr_page_image
from app.okf_spec import create_index_file
from app.utils import (
    concepts_dir,
    ensure_parent,
    file_sha256,
    images_dir,
    load_config,
    okf_bundle_dir,
    source_pdfs_dir,
    tables_dir,
)

logger = get_logger(__name__)

VERSION_RE = re.compile(r"\b[Vv](\d+(?:\.\d+)?)\b")
MIN_REFERENCE_STD = 3.0


def _extract_version(filename: str) -> str:
    matches = list(VERSION_RE.finditer(filename))
    if not matches:
        return ""
    return f"V{matches[-1].group(1)}"


def _render_okf(meta: dict[str, Any], body: str) -> str:
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"


def _grayscale_std(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    if not pixels:
        return 0.0
    mean = sum(pixels) / len(pixels)
    return (sum((p - mean) ** 2 for p in pixels) / len(pixels)) ** 0.5


def _dhash(img: Image.Image) -> int:
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class PageAwareIngestor:
    """Process every page; write manifests, asset registry, OKF, and chunk rows."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self._reference_hashes: list[int] = []
        self._load_reference_hashes()
        init_db()

    def _load_reference_hashes(self) -> None:
        ref_dir = Path(__file__).resolve().parent / "reference_images"
        if not ref_dir.is_dir():
            return
        for path in sorted(ref_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                img = Image.open(path)
                if _grayscale_std(img) < MIN_REFERENCE_STD:
                    continue
                self._reference_hashes.append(_dhash(img))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reference image load failed %s: %s", path, exc)

    def _skip_reason(self, img: Image.Image) -> str | None:
        if self.cfg.get("low_info_filter_enabled", True):
            if _grayscale_std(img) < float(self.cfg.get("low_info_max_std", 4.0)):
                return "blank"
        if self.cfg.get("brand_image_filter_enabled", True) and self._reference_hashes:
            max_ham = int(self.cfg.get("brand_image_max_hamming", 10))
            h = _dhash(img)
            for ref_h in self._reference_hashes:
                if _hamming(h, ref_h) <= max_ham:
                    return "brand"
        return None

    def _render_page(self, page: Any) -> Image.Image:
        dpi = int(self.cfg.get("page_render_dpi", 150))
        zoom = dpi / 72.0
        import fitz

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def ingest_pdf(self, pdf_path: Path | str, *, force: bool = False) -> dict[str, Any]:
        import fitz

        pdf_path = Path(pdf_path)
        checksum = file_sha256(pdf_path)
        pdf_name = pdf_path.name
        doc_name = pdf_path.stem
        version = _extract_version(pdf_name)
        registry = get_model_registry()

        with get_session() as session:
            existing = (
                session.query(Document)
                .filter_by(file_checksum=checksum)
                .order_by(Document.upload_timestamp.desc())
                .first()
            )
            if existing and existing.processing_status == "completed" and not force:
                logger.info("Duplicate checksum — skipping reprocessing: %s", pdf_name)
                return {
                    "document_id": existing.id,
                    "skipped": True,
                    "reason": "checksum_duplicate",
                    "report": existing.reports[-1].summary if existing.reports else None,
                }

            document = Document(
                name=doc_name,
                original_filename=pdf_name,
                file_checksum=checksum,
                file_size=pdf_path.stat().st_size,
                document_version=version,
                storage_path=str(pdf_path),
                processing_status="processing",
                access_scope=str(self.cfg.get("default_access_scope", "internal")),
                model_versions={
                    "embedding": registry.specs()["embedding"].name,
                    "vision": registry.specs()["vision"].name,
                    "ocr": registry.specs()["ocr"].name,
                },
            )
            session.add(document)
            session.flush()

            pdf = fitz.open(pdf_path)
            try:
                page_count = len(pdf)
                ensure_page_rows(session, document, page_count)
                out_dir = concepts_dir() / doc_name
                if force and out_dir.exists():
                    import shutil

                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                images_dir().mkdir(parents=True, exist_ok=True)
                tables_dir().mkdir(parents=True, exist_ok=True)

                page_types: list[PageType] = []
                page_image_occ_ids: dict[int, list[str]] = {}
                hash_pages: dict[str, set[int]] = {}
                # Pass 1: classify + collect image hashes for HF detection
                page_meta: list[dict[str, Any]] = []
                for idx in range(page_count):
                    page = pdf[idx]
                    page_no = idx + 1
                    signals = page_signals_from_pymupdf(page, page_no)
                    ptype = classify_page(signals)
                    page_types.append(ptype)
                    update_page(
                        session,
                        document.id,
                        page_no,
                        page_type=ptype.value,
                        text_extraction_status="pending",
                        ocr_status="pending",
                        image_extraction_status="pending",
                    )
                    page_meta.append({"signals": signals, "ptype": ptype, "page": page})

                document.document_type = classify_document(page_types).value

                band = float(self.cfg.get("header_footer_band_ratio", 0.15))
                hf_enabled = bool(self.cfg.get("header_footer_filter_enabled", True))
                min_pages = int(self.cfg.get("header_footer_min_pages", 2))
                index_hf = bool(self.cfg.get("index_header_footer_images", False))

                # Collect all images first for cross-page HF
                candidates_by_page: dict[int, list[dict[str, Any]]] = {}
                for idx in range(page_count):
                    page = pdf[idx]
                    page_no = idx + 1
                    page_h = float(page.rect.height) or 1.0
                    cands: list[dict[str, Any]] = []
                    for img_idx, img_info in enumerate(page.get_images(full=True) or []):
                        try:
                            xref = img_info[0]
                            pix = fitz.Pixmap(pdf, xref)
                            if pix.n > 4:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            mode = "RGBA" if pix.n == 4 else "RGB"
                            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                            reason = self._skip_reason(img)
                            if reason in {"blank", "brand"}:
                                continue
                            in_band = False
                            bbox = {}
                            try:
                                rects = page.get_image_rects(xref)
                                for r in rects:
                                    bbox = {
                                        "x0": r.x0,
                                        "y0": r.y0,
                                        "x1": r.x1,
                                        "y1": r.y1,
                                    }
                                    if r.y1 <= page_h * band or r.y0 >= page_h * (1.0 - band):
                                        in_band = True
                                        break
                            except Exception:  # noqa: BLE001
                                pass
                            ph = f"{_dhash(img):016x}"
                            hash_pages.setdefault(ph, set()).add(page_no)
                            cands.append(
                                {
                                    "img": img,
                                    "img_idx": img_idx,
                                    "phash": ph,
                                    "in_band": in_band,
                                    "bbox": bbox,
                                }
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Image extract fail %s p%s #%s: %s",
                                pdf_name,
                                page_no,
                                img_idx,
                                exc,
                            )
                    candidates_by_page[page_no] = cands

                hf_hashes = {
                    h
                    for h, pages in hash_pages.items()
                    if hf_enabled and len(pages) >= min_pages
                }

                all_chunk_ids: list[str] = []

                # Pass 2: per-page processing — never skip a page silently
                for idx in range(page_count):
                    page = pdf[idx]
                    page_no = idx + 1
                    ptype = page_types[idx]
                    content_count = 0
                    errors: list[str] = []
                    try:
                        text = (page.get_text("text") or "").strip()
                        chunks_for_page = []

                        # Text extraction
                        if text and len(text) >= 3:
                            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
                            chunks_for_page = pack_paragraphs(
                                paras or [text],
                                page_number=page_no,
                                section_heading="",
                                max_chars=1500,
                            )
                            update_page(
                                session,
                                document.id,
                                page_no,
                                text_extraction_status="ok",
                            )
                        else:
                            update_page(
                                session,
                                document.id,
                                page_no,
                                text_extraction_status="empty",
                            )

                        # OCR / page render for scanned or image-only
                        ocr_text = ""
                        if ptype in {PageType.SCANNED, PageType.IMAGE, PageType.EMPTY} or (
                            ptype == PageType.TEXT_IMAGE and len(text) < 40
                        ):
                            try:
                                rendered = self._render_page(page)
                                ocr = ocr_page_image(rendered)
                                ocr_text = ocr.text
                                if ocr_text:
                                    ocr_chunks = pack_paragraphs(
                                        [ocr_text],
                                        page_number=page_no,
                                        section_heading="OCR",
                                        chunk_type="ocr",
                                        max_chars=2000,
                                    )
                                    chunks_for_page.extend(ocr_chunks)
                                    # Also keep full-page image as asset
                                    asset, occ, _ = register_image(
                                        session,
                                        image=rendered,
                                        document_id=document.id,
                                        page_number=page_no,
                                        image_role="page_scan",
                                        indexable=True,
                                        caption=f"Page {page_no} scan",
                                        ocr_text=ocr_text,
                                    )
                                    page_image_occ_ids.setdefault(page_no, []).append(occ.id)
                                    content_count += 1
                                    update_page(
                                        session,
                                        document.id,
                                        page_no,
                                        ocr_status="ok" if ocr.confidence >= 0.4 else "low_confidence",
                                    )
                                else:
                                    # Still store rendered page for visual retrieval
                                    asset, occ, _ = register_image(
                                        session,
                                        image=rendered,
                                        document_id=document.id,
                                        page_number=page_no,
                                        image_role="page_render",
                                        indexable=True,
                                        caption=f"Page {page_no} render",
                                    )
                                    page_image_occ_ids.setdefault(page_no, []).append(occ.id)
                                    content_count += 1
                                    update_page(
                                        session,
                                        document.id,
                                        page_no,
                                        ocr_status="empty",
                                    )
                            except Exception as exc:  # noqa: BLE001
                                errors.append(f"ocr:{exc}")
                                update_page(
                                    session,
                                    document.id,
                                    page_no,
                                    ocr_status="failed",
                                    error=str(exc),
                                )
                        else:
                            update_page(
                                session,
                                document.id,
                                page_no,
                                ocr_status="n/a",
                            )

                        # Embedded images → asset registry (store once, keep occurrences)
                        for cand in candidates_by_page.get(page_no, []):
                            is_hf = cand["phash"] in hf_hashes or cand["in_band"]
                            role = "header_footer" if is_hf else "content"
                            indexable = (not is_hf) or index_hf
                            try:
                                asset, occ, created = register_image(
                                    session,
                                    image=cand["img"],
                                    document_id=document.id,
                                    page_number=page_no,
                                    image_role=role,
                                    indexable=indexable,
                                    caption="",
                                    bbox=cand.get("bbox") or {},
                                )
                                page_image_occ_ids.setdefault(page_no, []).append(occ.id)
                                content_count += 1
                                # Convenience symlink-style copy for UI under assets/images
                                if indexable:
                                    conv = images_dir() / f"{doc_name}_p{page_no}_fig{cand['img_idx']}.png"
                                    if not conv.exists():
                                        cand["img"].save(conv, format="PNG")
                                    rel = f"assets/images/{conv.name}"
                                    body = (
                                        f"# Figure p{page_no} fig{cand['img_idx']} - {doc_name}\n\n"
                                        f"![{conv.name}]({rel})\n\n"
                                        f"Caption: [NEEDS_CAPTION]\n\n"
                                        f"Asset: `{asset.id}` occurrence: `{occ.id}`\n"
                                        f"Page mapping: source=`{pdf_name}` page=`{page_no}`\n"
                                    )
                                    meta = {
                                        "type": "figure",
                                        "title": f"Figure p{page_no} fig{cand['img_idx']} - {doc_name}",
                                        "source": pdf_name,
                                        "source_loc": f"page:{page_no}",
                                        "tags": ["figure", doc_name, f"page-{page_no}", role],
                                        "business_unit": "general",
                                        "owner": "system",
                                        "related": [],
                                        "created": "",
                                        "confidentiality": "internal",
                                        "version": version,
                                        "section": "",
                                        "image_path": rel,
                                        "excluded": not indexable,
                                        "image_role": role,
                                        "page": page_no,
                                        "global_image_asset_id": asset.id,
                                        "occurrence_id": occ.id,
                                        "content_hash": asset.content_hash,
                                    }
                                    from datetime import date

                                    meta["created"] = str(date.today())
                                    fig_path = out_dir / f"figure_p{page_no}_{cand['img_idx']}.md"
                                    ensure_parent(fig_path)
                                    fig_path.write_text(_render_okf(meta, body), encoding="utf-8")
                            except Exception as exc:  # noqa: BLE001
                                errors.append(f"image:{exc}")

                        update_page(
                            session,
                            document.id,
                            page_no,
                            image_extraction_status="ok",
                            duplicate_detection_status="ok",
                            table_extraction_status="n/a",
                            diagram_analysis_status="n/a",
                        )

                        # Associate images and persist chunks
                        chunks_for_page = associate_images_to_chunks(
                            chunks_for_page, page_image_occ_ids
                        )
                        page_chunk_ids: list[str] = []
                        for draft in chunks_for_page:
                            # Skip repetitive short headers (simple heuristic)
                            if draft.chunk_type == "text" and len(draft.text) < 40:
                                # still record but mark non-indexable if looks like page number only
                                if re.fullmatch(r"\d+", draft.text.strip()):
                                    continue
                            rec = ChunkRecord(
                                document_id=document.id,
                                page_number=draft.page_number,
                                section_heading=draft.section_heading,
                                parent_section=draft.parent_section,
                                chunk_type=draft.chunk_type,
                                chunk_text=draft.text[:12000],
                                content_hash=draft.content_hash,
                                associated_image_ids=draft.associated_image_ids,
                                associated_table_ids=draft.associated_table_ids,
                                source_coordinates=draft.source_coordinates,
                                indexable=True,
                            )
                            session.add(rec)
                            session.flush()
                            page_chunk_ids.append(rec.id)
                            all_chunk_ids.append(rec.id)
                            content_count += 1

                            # OKF text concept
                            text_hash = draft.content_hash[:10]
                            body = draft.text[:8000]
                            if len(body.strip()) <= 20:
                                body = body + f"\n\n(Extracted from {pdf_name}, page {page_no})"
                            meta = {
                                "type": "document",
                                "title": body.replace("\n", " ").strip()[:80],
                                "source": pdf_name,
                                "source_loc": f"page:{page_no}",
                                "tags": ["text", draft.chunk_type, doc_name, f"page-{page_no}"],
                                "business_unit": "general",
                                "owner": "system",
                                "related": draft.associated_image_ids,
                                "created": "",
                                "confidentiality": "internal",
                                "version": version,
                                "section": draft.section_heading,
                                "chunk_id": rec.id,
                                "document_id": document.id,
                                "content_hash": draft.content_hash,
                            }
                            from datetime import date

                            meta["created"] = str(date.today())
                            okf_path = out_dir / f"text_p{page_no}_{text_hash}.md"
                            ensure_parent(okf_path)
                            okf_path.write_text(_render_okf(meta, body), encoding="utf-8")
                            rec.okf_path = str(okf_path.relative_to(okf_bundle_dir())).replace("\\", "/")

                        # Link prev/next
                        for (prev_id, next_id), cid in zip(
                            link_chunks(page_chunk_ids), page_chunk_ids
                        ):
                            row = session.get(ChunkRecord, cid)
                            if row:
                                row.previous_chunk_id = prev_id
                                row.next_chunk_id = next_id

                        update_page(
                            session,
                            document.id,
                            page_no,
                            chunking_status="ok" if page_chunk_ids or content_count else "empty",
                            extracted_content_count=content_count,
                        )

                        # Outcome
                        if errors and content_count == 0:
                            outcome = "failed"
                        elif content_count == 0:
                            outcome = "empty"
                        elif ocr_text and not text:
                            outcome = "ocr_ok"
                        elif page_image_occ_ids.get(page_no) and not text:
                            outcome = "image_ok"
                        else:
                            outcome = "text_ok"
                        update_page(
                            session,
                            document.id,
                            page_no,
                            outcome=outcome,
                            error="; ".join(errors) if errors else None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Page %s failed for %s", page_no, pdf_name)
                        update_page(
                            session,
                            document.id,
                            page_no,
                            outcome="failed",
                            error=str(exc),
                        )

                report = build_completeness_report(session, document)
                create_index_file(okf_bundle_dir())
                logger.info(
                    "Pipeline ingest %s complete=%s pages=%s type=%s",
                    pdf_name,
                    report.get("complete"),
                    page_count,
                    document.document_type,
                )
                return {
                    "document_id": document.id,
                    "skipped": False,
                    "report": report,
                    "document_type": document.document_type,
                    "chunks": len(all_chunk_ids),
                }
            finally:
                pdf.close()

    def ingest_all(self, *, force: bool = False) -> dict[str, Any]:
        src = source_pdfs_dir()
        src.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(src.glob("*.pdf"))
        results = []
        for pdf in pdfs:
            try:
                results.append(self.ingest_pdf(pdf, force=force))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed ingest %s: %s", pdf.name, exc)
                results.append({"file": pdf.name, "error": str(exc)})
        return {"processed": len(results), "results": results}
