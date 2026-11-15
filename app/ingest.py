"""PDF → OKF concept files (tables, figures, text sections)."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from app.logger import get_logger
from app.okf_spec import create_index_file
from app.utils import (
    concepts_dir,
    ensure_parent,
    images_dir,
    load_config,
    okf_bundle_dir,
    source_pdfs_dir,
    tables_dir,
)

logger = get_logger(__name__)

MIN_REFERENCE_STD = 3.0
VERSION_RE = re.compile(r"\b[Vv](\d+(?:\.\d+)?)\b")


def _render_okf_content(meta: dict[str, Any], body: str) -> str:
    """Serialize frontmatter + body via yaml.safe_dump (never hand-rolled)."""
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"


def _extract_version(filename: str) -> str:
    """Rightmost V/v version marker in the PDF filename → 'V1.0' or ''."""
    matches = list(VERSION_RE.finditer(filename))
    if not matches:
        return ""
    m = matches[-1]
    return f"V{m.group(1)}"


def _item_page_no(item: Any, default: int = 1) -> int:
    """Best-effort page number from a docling layout item."""
    page = getattr(item, "page_no", None)
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    prov = getattr(item, "prov", None)
    if prov:
        try:
            return int(prov[0].page_no)
        except (IndexError, AttributeError, TypeError, ValueError):
            pass
    return default


def _dhash(img: Image.Image) -> int:
    """64-bit perceptual difference hash (8×9 grayscale, Lanczos)."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _grayscale_std(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    if not pixels:
        return 0.0
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return variance ** 0.5


class EnterpriseOKFIngestor:
    """Extract tables, images, and text sections from PDFs into an OKF bundle."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self._converter = None
        self._reference_hashes: list[int] = []
        self._load_reference_hashes()

    def _get_converter(self):
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF]
            )
        return self._converter

    def _load_reference_hashes(self) -> None:
        ref_dir = Path(__file__).resolve().parent / "reference_images"
        if not ref_dir.is_dir():
            return
        for path in sorted(ref_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                img = Image.open(path)
                std = _grayscale_std(img)
                if std < MIN_REFERENCE_STD:
                    logger.warning(
                        "Reference image %s refused (grayscale std %.2f < %.1f) "
                        "— perceptual hash would be degenerate",
                        path.name,
                        std,
                        MIN_REFERENCE_STD,
                    )
                    continue
                self._reference_hashes.append(_dhash(img))
                logger.info("Loaded brand reference hash from %s", path.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load reference image %s: %s", path, exc)

    def _skip_reason(self, img: Image.Image) -> str | None:
        """Return skip reason ('blank'/'brand') or None if image should be kept."""
        cfg = self.cfg
        if cfg.get("low_info_filter_enabled", True):
            max_std = float(cfg.get("low_info_max_std", 4.0))
            if _grayscale_std(img) < max_std:
                return "blank"

        if cfg.get("brand_image_filter_enabled", True) and self._reference_hashes:
            max_ham = int(cfg.get("brand_image_max_hamming", 10))
            h = _dhash(img)
            for ref_h in self._reference_hashes:
                if _hamming_distance(h, ref_h) <= max_ham:
                    return "brand"
        return None

    def _base_meta(
        self,
        *,
        ctype: str,
        title: str,
        source: str,
        source_loc: str,
        tags: list[str],
        version: str,
        section: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "type": ctype,
            "title": title,
            "source": source,
            "source_loc": source_loc,
            "tags": tags,
            "business_unit": "general",
            "owner": "system",
            "related": [],
            "created": str(date.today()),
            "confidentiality": "internal",
            "version": version,
            "section": section,
        }
        if extra:
            meta.update(extra)
        return meta

    def _write_concept(self, path: Path, meta: dict[str, Any], body: str) -> None:
        ensure_parent(path)
        path.write_text(_render_okf_content(meta, body), encoding="utf-8")

    # ------------------------------------------------------------------ tables
    def _extract_tables(
        self, doc: Any, doc_name: str, pdf_name: str, version: str, out_dir: Path
    ) -> int:
        count = 0
        tables = getattr(doc, "tables", None) or []
        for idx, table in enumerate(tables):
            try:
                page_no = _item_page_no(table)
                md_table = ""
                json_records = "[]"
                try:
                    df = table.export_to_dataframe()
                    md_table = df.to_markdown(index=False)
                    json_records = df.to_json(orient="records")
                except Exception:  # noqa: BLE001
                    try:
                        md_table = table.export_to_markdown()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Table %d export failed for %s: %s", idx, pdf_name, exc
                        )
                        md_table = ""

                sidecar = tables_dir() / f"{doc_name}_table_{idx}.json"
                ensure_parent(sidecar)
                sidecar.write_text(json_records or "[]", encoding="utf-8")
                rel_sidecar = f"assets/tables/{sidecar.name}"

                body = (
                    f"# Table {idx} - {doc_name}\n\n"
                    f"{md_table}\n\n"
                    f"## Machine Readable\n\n"
                    f"JSON sidecar: `{rel_sidecar}`\n"
                )
                meta = self._base_meta(
                    ctype="table",
                    title=f"Table {idx} - {doc_name}",
                    source=pdf_name,
                    source_loc=f"page:{page_no}",
                    tags=["table", doc_name, f"page-{page_no}"],
                    version=version,
                    section="",
                )
                out_path = out_dir / f"table_p{page_no}_{idx}.md"
                self._write_concept(out_path, meta, body)
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping table %d in %s due to error: %s", idx, pdf_name, exc
                )
        return count

    # ------------------------------------------------------------------ images
    def _extract_images(
        self, pdf_path: Path, doc_name: str, pdf_name: str, version: str, out_dir: Path
    ) -> tuple[int, int, int, int]:
        """Extract images; drop blank/brand and cross-page header/footer repeats.

        Returns ``(kept, skipped_brand, skipped_blank, skipped_header_footer)``.
        """
        import fitz  # PyMuPDF — local import so unit tests need no native DLL

        kept = skipped_brand = skipped_blank = skipped_hf = 0
        try:
            pdf = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyMuPDF open failed for %s: %s", pdf_path, exc)
            return 0, 0, 0, 0

        # Pass 1: decode candidates and count perceptual hashes across pages
        candidates: list[dict[str, Any]] = []
        hash_pages: dict[int, set[int]] = {}
        band = float(self.cfg.get("header_footer_band_ratio", 0.15))
        hf_enabled = bool(self.cfg.get("header_footer_filter_enabled", True))
        min_pages = int(self.cfg.get("header_footer_min_pages", 3))

        try:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                page_h = float(page.rect.height) or 1.0
                images = page.get_images(full=True)
                for img_idx, img_info in enumerate(images):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(pdf, xref)
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        mode = "RGBA" if pix.n == 4 else "RGB"
                        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

                        reason = self._skip_reason(img)
                        if reason == "blank":
                            skipped_blank += 1
                            continue
                        if reason == "brand":
                            skipped_brand += 1
                            continue

                        page_display = page_num + 1
                        h = _dhash(img)
                        in_band = False
                        try:
                            rects = page.get_image_rects(xref)
                            for r in rects:
                                if r.y1 <= page_h * band or r.y0 >= page_h * (1.0 - band):
                                    in_band = True
                                    break
                        except Exception:  # noqa: BLE001
                            in_band = False

                        hash_pages.setdefault(h, set()).add(page_display)
                        candidates.append(
                            {
                                "img": img,
                                "page": page_display,
                                "img_idx": img_idx,
                                "dhash": h,
                                "in_band": in_band,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Skipping image %d on page %d of %s: %s",
                            img_idx,
                            page_num + 1,
                            pdf_name,
                            exc,
                        )
        finally:
            pdf.close()

        # Count total occurrences of each hash (any page)
        hash_counts: dict[int, int] = {}
        for cand in candidates:
            hash_counts[cand["dhash"]] = hash_counts.get(cand["dhash"], 0) + 1

        # Cross-page repeats (headers/footers) and any duplicate hash → never load
        drop_dupes = bool(self.cfg.get("drop_duplicate_images", True))
        repeat_hashes = {
            h
            for h, pages in hash_pages.items()
            if hf_enabled and len(pages) >= min_pages
        }
        duplicate_hashes = {
            h for h, count in hash_counts.items() if drop_dupes and count > 1
        }
        banned_hashes = repeat_hashes | duplicate_hashes

        # Pass 2: persist only unique, non-repeating content images
        for cand in candidates:
            h = cand["dhash"]
            if h in banned_hashes:
                skipped_hf += 1
                continue

            page_display = cand["page"]
            img_idx = cand["img_idx"]
            img = cand["img"]
            img_name = f"{doc_name}_p{page_display}_fig{img_idx}.png"
            img_path = images_dir() / img_name
            ensure_parent(img_path)
            img.save(img_path, format="PNG")
            rel_img = f"assets/images/{img_name}"

            body = (
                f"# Figure p{page_display} fig{img_idx} - {doc_name}\n\n"
                f"![{img_name}]({rel_img})\n\n"
                f"Caption: [NEEDS_CAPTION]\n\n"
                f"Page mapping: source=`{pdf_name}` page=`{page_display}`\n"
            )
            meta = self._base_meta(
                ctype="figure",
                title=f"Figure p{page_display} fig{img_idx} - {doc_name}",
                source=pdf_name,
                source_loc=f"page:{page_display}",
                tags=["figure", doc_name, f"page-{page_display}"],
                version=version,
                section="",
                extra={
                    "image_path": rel_img,
                    "excluded": False,
                    "image_role": "content",
                    "page": page_display,
                },
            )
            out_path = out_dir / f"figure_p{page_display}_{img_idx}.md"
            self._write_concept(out_path, meta, body)
            kept += 1

        logger.info(
            "Image dedupe for %s: candidates=%d unique_kept=%d "
            "banned_duplicate_hashes=%d banned_header_footer_hashes=%d",
            pdf_name,
            len(candidates),
            kept,
            len(duplicate_hashes),
            len(repeat_hashes),
        )
        return kept, skipped_brand, skipped_blank, skipped_hf
    # ----------------------------------------------------------- text sections
    def _extract_text_sections(
        self, doc: Any, doc_name: str, pdf_name: str, version: str, out_dir: Path
    ) -> int:
        count = 0
        heading_trail: list[str] = []
        list_buffer: list[tuple[str, int, str]] = []  # (text, page, label)

        def flush_list_buffer() -> None:
            nonlocal count
            if not list_buffer:
                return
            combined = "\n".join(t for t, _, _ in list_buffer)
            page_no = list_buffer[0][1]
            section_label = list_buffer[0][2] or (
                heading_trail[-1] if heading_trail else ""
            )
            context = " > ".join(heading_trail) if heading_trail else ""
            title_src = combined if combined.strip() else (
                heading_trail[-1] if heading_trail else "list"
            )
            title = title_src.replace("\n", " ").strip()[:80]
            body_parts = []
            if context:
                body_parts.append(f"Context: {context}")
            body_parts.append(combined[:8000])
            body = "\n\n".join(body_parts)
            text_hash = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:10]
            meta = self._base_meta(
                ctype="document",
                title=title,
                source=pdf_name,
                source_loc=f"page:{page_no}",
                tags=["text", "list_item", doc_name, f"page-{page_no}"],
                version=version,
                section=section_label,
            )
            out_path = out_dir / f"text_p{page_no}_{text_hash}.md"
            self._write_concept(out_path, meta, body)
            count += 1
            list_buffer.clear()

        texts = getattr(doc, "texts", None) or []
        for item in texts:
            try:
                text = (getattr(item, "text", None) or "").strip()
                raw_label = getattr(item, "label", None)
                if raw_label is None:
                    label = ""
                elif hasattr(raw_label, "value"):
                    label = str(raw_label.value).lower()
                else:
                    label = str(raw_label).lower()

                page_no = _item_page_no(item)
                is_heading = label in {"section_header", "title"}
                is_list = label == "list_item"

                if is_heading:
                    flush_list_buffer()
                    if text:
                        heading_trail.append(text)
                    # Headings are never dropped for being short
                    section_label = text or (
                        heading_trail[-1] if heading_trail else ""
                    )
                    title = text.replace("\n", " ").strip()[:80] or "Heading"
                    body = text[:8000] if text else section_label
                    if len(body) < 3 and not is_heading:
                        continue
                    # Ensure body > 20 chars for OKF validation by padding with context
                    if len(body.strip()) <= 20:
                        pad = f"\n\n(Section heading from {pdf_name}, page {page_no})"
                        body = body + pad
                    text_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
                    tag_label = (
                        "section_header"
                        if label == "section_header"
                        else "title"
                        if label == "title"
                        else "heading"
                    )
                    meta = self._base_meta(
                        ctype="document",
                        title=title,
                        source=pdf_name,
                        source_loc=f"page:{page_no}",
                        tags=["text", tag_label, doc_name, f"page-{page_no}"],
                        version=version,
                        section=section_label,
                    )
                    out_path = out_dir / f"text_p{page_no}_{text_hash}.md"
                    self._write_concept(out_path, meta, body)
                    count += 1
                    continue

                if is_list:
                    if len(text) < 3:
                        continue
                    section_label = heading_trail[-1] if heading_trail else ""
                    list_buffer.append((text, page_no, section_label))
                    continue

                # Non-heading, non-list item
                flush_list_buffer()
                if len(text) < 3:
                    continue

                context = " > ".join(heading_trail) if heading_trail else ""
                section_label = heading_trail[-1] if heading_trail else ""
                title = text.replace("\n", " ").strip()[:80]
                body_parts = []
                if context:
                    body_parts.append(f"Context: {context}")
                body_parts.append(text[:8000])
                body = "\n\n".join(body_parts)
                text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
                meta = self._base_meta(
                    ctype="document",
                    title=title,
                    source=pdf_name,
                    source_loc=f"page:{page_no}",
                    tags=["text", doc_name, f"page-{page_no}"],
                    version=version,
                    section=section_label,
                )
                out_path = out_dir / f"text_p{page_no}_{text_hash}.md"
                self._write_concept(out_path, meta, body)
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping text item in %s due to error: %s", pdf_name, exc
                )

        flush_list_buffer()
        return count

    # --------------------------------------------------------------- lifecycle
    def _clear_doc_artifacts(self, doc_name: str, out_dir: Path) -> None:
        """Remove prior concept/image/table artifacts for a document."""
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for img in images_dir().glob(f"{doc_name}_p*_fig*.png"):
            img.unlink(missing_ok=True)
        for tbl in tables_dir().glob(f"{doc_name}_table_*.json"):
            tbl.unlink(missing_ok=True)

    def _extract_text_pymupdf(
        self, pdf_path: Path, doc_name: str, pdf_name: str, version: str, out_dir: Path
    ) -> int:
        """Fallback text chunking via PyMuPDF when docling is unavailable."""
        import fitz

        count = 0
        pdf = fitz.open(pdf_path)
        try:
            for page_idx in range(len(pdf)):
                page_no = page_idx + 1
                raw = (pdf[page_idx].get_text("text") or "").strip()
                if len(raw) < 3:
                    continue
                # Split into paragraph-ish blocks, then pack into ~1500-char chunks
                paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
                if not paras:
                    paras = [raw]
                buf = ""
                for para in paras:
                    candidate = f"{buf}\n\n{para}".strip() if buf else para
                    if len(candidate) > 1500 and buf:
                        count += self._write_text_chunk(
                            buf, page_no, doc_name, pdf_name, version, out_dir
                        )
                        buf = para
                    else:
                        buf = candidate
                if buf and len(buf.strip()) >= 3:
                    count += self._write_text_chunk(
                        buf, page_no, doc_name, pdf_name, version, out_dir
                    )
        finally:
            pdf.close()
        return count

    def _write_text_chunk(
        self,
        text: str,
        page_no: int,
        doc_name: str,
        pdf_name: str,
        version: str,
        out_dir: Path,
    ) -> int:
        body = text[:8000]
        if len(body.strip()) <= 20:
            body = body + f"\n\n(Extracted from {pdf_name}, page {page_no})"
        title = body.replace("\n", " ").strip()[:80]
        text_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
        meta = self._base_meta(
            ctype="document",
            title=title,
            source=pdf_name,
            source_loc=f"page:{page_no}",
            tags=["text", "pymupdf", doc_name, f"page-{page_no}"],
            version=version,
            section="",
        )
        out_path = out_dir / f"text_p{page_no}_{text_hash}.md"
        self._write_concept(out_path, meta, body)
        return 1

    def ingest_pdf(self, pdf_path: Path | str, force: bool = False) -> bool:
        """Ingest a single PDF into the OKF bundle. Returns True if processed."""
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.name
        doc_name = pdf_path.stem
        out_dir = concepts_dir() / doc_name

        if out_dir.exists() and any(out_dir.iterdir()) and not force:
            logger.info("already ingested, skipping: %s", pdf_name)
            return False

        logger.info("Ingesting %s (force=%s)", pdf_name, force)
        version = _extract_version(pdf_name)
        doc = None
        used_fallback = False

        # Prefer docling; fall back to PyMuPDF if models/HF/auth fail
        try:
            converter = self._get_converter()
            result = converter.convert(str(pdf_path))
            doc = result.document
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Docling failed for %s (%s). Falling back to PyMuPDF text+image extract.",
                pdf_name,
                exc,
            )
            used_fallback = True

        if force or (out_dir.exists() and any(out_dir.iterdir())):
            self._clear_doc_artifacts(doc_name, out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir().mkdir(parents=True, exist_ok=True)
        tables_dir().mkdir(parents=True, exist_ok=True)

        if used_fallback or doc is None:
            n_tables = 0
            n_text = self._extract_text_pymupdf(
                pdf_path, doc_name, pdf_name, version, out_dir
            )
        else:
            n_tables = self._extract_tables(doc, doc_name, pdf_name, version, out_dir)
            n_text = self._extract_text_sections(
                doc, doc_name, pdf_name, version, out_dir
            )

        n_figs, n_brand, n_blank, n_hf = self._extract_images(
            pdf_path, doc_name, pdf_name, version, out_dir
        )

        logger.info(
            "Ingested %s → tables=%d figures=%d "
            "(skipped brand=%d blank=%d header_footer=%d) text=%d (fallback=%s)",
            pdf_name,
            n_tables,
            n_figs,
            n_brand,
            n_blank,
            n_hf,
            n_text,
            used_fallback,
        )
        return True

    def ingest_all(self, force: bool = False) -> int:
        """Ingest every PDF under source_pdfs/. Returns count of processed docs."""
        src = source_pdfs_dir()
        src.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(src.glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found in %s", src)
            return 0

        processed = 0
        for pdf in pdfs:
            try:
                if self.ingest_pdf(pdf, force=force):
                    processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest %s: %s", pdf.name, exc)

        create_index_file(okf_bundle_dir())
        logger.info("Ingest complete: %d/%d PDFs processed", processed, len(pdfs))
        return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs from source_pdfs/ into the OKF bundle."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if concepts already exist (destructive after successful convert).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy Docling-only ingestor instead of page-aware pipeline.",
    )
    args = parser.parse_args()
    if args.legacy:
        EnterpriseOKFIngestor().ingest_all(force=args.force)
    else:
        from app.pipeline import PageAwareIngestor

        PageAwareIngestor().ingest_all(force=args.force)


if __name__ == "__main__":
    main()
