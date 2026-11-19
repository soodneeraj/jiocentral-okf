"""Document-faithful procedure reconstruction: ordered steps + interleaved images."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.logger import get_logger
from app.utils import load_config, okf_bundle_dir

logger = get_logger(__name__)

_STEP_RE = re.compile(
    r"(?im)^(?:\s*)(?:step\s+\d+[.:)]|\d+[.)]\s+\S|(?:fig(?:ure)?\.?\s*[\d.]+))",
)
_FIG_REF_RE = re.compile(
    r"(?i)\b(?:fig(?:ure)?\.?\s*)([\d]+(?:\.[\d]+)*)\b",
)
_STEP_SPLIT_RE = re.compile(
    r"(?im)(?=^Step\s+\d+[.:)]|^\d+[.)]\s+)",
)


def looks_like_procedure(text: str) -> bool:
    """True when body looks like a step-by-step procedure."""
    if not text:
        return False
    body = text.strip()
    step_hits = len(re.findall(r"(?i)\bstep\s+\d+", body))
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+\S", body))
    fig_hits = len(_FIG_REF_RE.findall(body))
    return step_hits >= 1 or numbered >= 2 or (fig_hits >= 1 and step_hits + numbered >= 1)


def query_wants_procedure(query: str) -> bool:
    q = (query or "").lower()
    keys = (
        "step",
        "steps",
        "how to",
        "how do i",
        "procedure",
        "instructions",
        "walkthrough",
        "follow",
        "perform",
        "configure",
        "create",
        "insert",
        "show me how",
    )
    return any(k in q for k in keys)


def extract_figure_refs(text: str) -> list[str]:
    """Return ordered unique figure labels like '1.10' from body text."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _FIG_REF_RE.finditer(text or ""):
        label = m.group(1).strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "be",
        "by",
        "from",
        "as",
        "at",
        "it",
        "this",
        "that",
        "what",
        "how",
        "do",
        "i",
        "me",
        "my",
        "please",
        "show",
        "tell",
        "explain",
        "give",
        "list",
        "steps",
        "step",
        "procedure",
    }
)


def query_keywords(query: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9_\-/']{1,}", (query or "").lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def score_body_for_query(body: str, query: str) -> float:
    """Higher = better topical match between query and passage."""
    if not body:
        return -1.0
    text = body.lower()
    score = 0.0
    for fig in extract_figure_refs(query):
        if re.search(rf"(?i)\bfig(?:ure)?\.?\s*{re.escape(fig)}\b", body):
            score += 8.0
    for kw in query_keywords(query):
        # whole-word-ish count, capped
        hits = len(re.findall(rf"(?i)\b{re.escape(kw)}\b", text))
        if hits:
            score += min(hits, 3) * (2.5 if len(kw) >= 5 else 1.2)
    if re.search(r"(?i)steps?\s+to\s+follow", text):
        score += 1.0
    # Prefer passages that start near the ask (dispatch inbox etc.)
    head = text[:400]
    for kw in query_keywords(query)[:6]:
        if re.search(rf"(?i)\b{re.escape(kw)}\b", head):
            score += 1.5
    return score


def focus_body_on_query(body: str, query: str) -> str:
    """
    Crop a page body to the section that best answers the query.

    Keeps STEPS / figure / keyword-aligned region so neighboring topics
    from the same page (or chunk) are not dumped into the answer.
    """
    text = (body or "").strip()
    if not text or not (query or "").strip():
        return text

    figs = extract_figure_refs(query)
    kws = query_keywords(query)
    candidates: list[int] = []
    qlow = (query or "").lower()

    for fig in figs:
        for m in re.finditer(rf"(?i)\bfig(?:ure)?\.?\s*{re.escape(fig)}\b", text):
            candidates.append(m.start())

    # Prefer multi-word topic anchors (e.g. "copy and paste", "clipboard")
    phrase_res: list[re.Pattern[str]] = []
    if "copy" in kws and "paste" in kws:
        phrase_res.append(re.compile(r"(?i)copy\s+and\s+paste|clipboard"))
    if "drag" in kws and "drop" in kws:
        phrase_res.append(re.compile(r"(?i)drag\s+and\s+drop"))
    if "practical" in qlow and "activity" in qlow:
        phrase_res.append(re.compile(r"(?i)practical\s+activity"))
    for pre in phrase_res:
        for m in pre.finditer(text):
            candidates.append(m.start())

    # If the query names a figure or phrase, prefer that over generic keyword hits
    if not candidates:
        for kw in kws:
            for m in re.finditer(rf"(?i)\b{re.escape(kw)}\b", text):
                candidates.append(m.start())
                break  # first hit per keyword is enough
    else:
        pass

    # Prefer a STEPS TO FOLLOW block near the best candidate (plain or markdown-wrapped)
    _steps_re = re.compile(
        r"(?im)^(?:\s*)(?:\*\*|__|<u>|<strong>|<em>)*\s*STEPS?\s+TO\s+FOLLOW\s*:?\s*(?:\*\*|__|</u>|</strong>|</em>)*\s*$"
    )
    steps_iter = list(_steps_re.finditer(text))
    start = 0
    if candidates:
        anchor = min(candidates)
        # Walk back to nearest STEPS heading before anchor, else paragraph start
        chosen_steps = None
        for sm in steps_iter:
            if sm.start() <= anchor:
                chosen_steps = sm
            else:
                break
        if chosen_steps is not None:
            start = chosen_steps.start()
        else:
            # paragraph / line start before anchor
            nl = text.rfind("\n", 0, max(0, anchor - 1))
            start = 0 if nl < 0 else nl + 1
            # include a bit of lead-in (up to 1 prior short paragraph)
            prev = text.rfind("\n\n", 0, start)
            if prev >= 0 and start - prev < 280:
                start = prev + 2
    elif steps_iter:
        # No keyword hits — take first STEPS block only (still better than whole page)
        start = steps_iter[0].start()

    cropped = text[start:].strip()

    # Stop before the next unrelated STEPS block if present after some content
    later_steps = list(_steps_re.finditer(cropped))
    if len(later_steps) >= 2:
        # Keep from first STEPS through just before the second STEPS
        cropped = cropped[: later_steps[1].start()].strip()

    # Drop running headers/footers noise
    cropped = re.sub(r"(?im)^Ver\.?\s*\d+(?:\.\d+)?\s*\[[^\]]*\]\s*$", "", cropped)
    cropped = re.sub(r"(?im)^\d{1,4}\s*$", "", cropped)
    cropped = re.sub(r"\n{3,}", "\n\n", cropped).strip()

    # If crop somehow emptied, fall back
    return cropped or text


def needs_continuation_page(body: str, query: str = "") -> bool:
    """True when the focused excerpt likely continues on the next page."""
    text = body or ""
    if not text.strip():
        return False
    qlow = (query or "").lower()
    # Practical activities / step lists often span two pages
    if re.search(r"(?i)practical\s+activity", text):
        step_nums = [int(n) for n in re.findall(r"(?im)^\s*(?:\*\*)?Step\s+(\d+)", text)]
        if not step_nums or max(step_nums) <= 3:
            return True
    if re.search(r"(?i)as shown in\s+fig", text) and re.search(
        r"(?im)^\s*(?:\*\*)?Step\s+\d+", text
    ):
        return True
    # Copy/paste answers frequently continue with the practical activity on next page
    if ("copy" in qlow and "paste" in qlow) or "clipboard" in qlow:
        if re.search(r"(?i)practical\s+activity|step\s+\d+", text):
            return True
    return False


def select_focused_hits(
    hits: list[dict[str, Any]],
    query: str,
    *,
    max_pages: int = 1,
) -> list[dict[str, Any]]:
    """Keep only the best-matching page(s) for a concise answer."""
    if not hits:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for h in hits:
        body = str(h.get("body") or "")
        scored.append((score_body_for_query(body, query), h))
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score = scored[0][0]
    # Always keep the top hit; add others only if close in score and different page
    kept: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for score, h in scored:
        page = int(h.get("page") or 0)
        if kept and score < best_score * 0.55 and score < best_score - 4:
            break
        if page in seen_pages:
            continue
        kept.append(h)
        seen_pages.add(page)
        if len(kept) >= max(1, max_pages):
            break
    # Preserve document order among the focused pages
    kept.sort(key=lambda h: int(h.get("page") or 0))
    return kept


def _fig_sort_key(img: dict[str, Any]) -> tuple[int, str]:
    path = str(img.get("image_path") or "")
    m = re.search(r"fig(\d+)", path, re.I)
    idx = int(m.group(1)) if m else 9999
    return (idx, path)


def sort_images_for_page(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(images, key=_fig_sort_key)


def match_image_for_fig_ref(
    fig_label: str,
    page_images: list[dict[str, Any]],
    used: set[str],
) -> dict[str, Any] | None:
    """Prefer caption/figure_label match; else next unused figN in order."""
    needle = fig_label.strip().lower()
    for img in page_images:
        ip = str(img.get("image_path") or "")
        if not ip or ip in used:
            continue
        label = str(img.get("figure_label") or "").strip().lower()
        caption = str(img.get("caption") or img.get("image_description") or "").lower()
        title = str(img.get("title") or "").lower()
        blob = f"{label} {caption} {title}"
        if needle and (
            label == needle
            or f"fig. {needle}" in blob
            or f"figure {needle}" in blob
            or f"fig {needle}" in blob
        ):
            return img
    for img in sort_images_for_page(page_images):
        ip = str(img.get("image_path") or "")
        if ip and ip not in used:
            return img
    return None


def _markdown_image(img: dict[str, Any]) -> str:
    ip = str(img.get("image_path") or "").replace("\\", "/")
    # Parentheses in filenames break markdown ![alt](path) parsers — encode them
    ip_md = ip.replace("(", "%28").replace(")", "%29")
    caption = str(img.get("caption") or img.get("figure_label") or "").strip()
    alt = caption or Path_name(ip)
    # Keep alt free of ']' which would also break markdown
    alt = alt.replace("]", "")
    label = str(img.get("figure_label") or "").strip()
    lines = [f"![{alt}]({ip_md})"]
    if label:
        lines.append(f"*Fig. {label}*")
    elif caption and not caption.lower().startswith("figure from"):
        lines.append(f"*{caption}*")
    return "\n".join(lines)


def Path_name(ip: str) -> str:
    return Path(ip).name if ip else "figure"


def _load_okf_figure_index() -> dict[str, dict[str, Any]]:
    """Map image_path → figure meta from OKF bundle (includes page/source)."""
    try:
        from app.okf_spec import validate_okf_file
    except Exception:  # noqa: BLE001
        return {}
    bundle = okf_bundle_dir()
    concepts = bundle / "concepts"
    by_path: dict[str, dict[str, Any]] = {}
    if not concepts.exists():
        return by_path
    for md in concepts.rglob("figure_*.md"):
        try:
            meta, _body = validate_okf_file(md)
        except ValueError:
            continue
        if meta.get("excluded") is True:
            continue
        ip = str(meta.get("image_path") or "").replace("\\", "/")
        if not ip:
            continue
        page = int(meta.get("page") or 0)
        if not page:
            loc = str(meta.get("source_loc") or "")
            if "page:" in loc:
                try:
                    page = int(loc.split("page:", 1)[1].split()[0])
                except ValueError:
                    page = 0
        row = dict(meta)
        row["page"] = page
        row["image_path"] = ip
        by_path[ip] = row
    return by_path


def overlay_figure_labels_from_okf(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill missing figure_label/caption from OKF figure files when Milvus lags."""
    if not images:
        return images
    by_path = _load_okf_figure_index()
    out: list[dict[str, Any]] = []
    for img in images:
        row = dict(img)
        ip = str(row.get("image_path") or "").replace("\\", "/")
        meta = by_path.get(ip)
        if meta:
            if not row.get("figure_label") and meta.get("figure_label"):
                row["figure_label"] = meta.get("figure_label")
            cap = str(meta.get("caption") or "")
            if cap and (
                not row.get("caption")
                or str(row.get("caption") or "").lower().startswith("figure from")
            ):
                row["caption"] = cap
        out.append(row)
    return out


def load_okf_images_for_pages(
    source: str,
    pages: list[int],
) -> list[dict[str, Any]]:
    """Filesystem fallback when Milvus image collection is empty/rebuilding."""
    if not source or not pages:
        return []
    want = {int(p) for p in pages if int(p) > 0}
    out: list[dict[str, Any]] = []
    for meta in _load_okf_figure_index().values():
        if str(meta.get("source") or "") != source:
            continue
        if int(meta.get("page") or 0) not in want:
            continue
        out.append(
            {
                "type": "figure",
                "title": meta.get("title"),
                "source": meta.get("source"),
                "source_loc": meta.get("source_loc"),
                "page": int(meta.get("page") or 0),
                "image_path": meta.get("image_path"),
                "caption": meta.get("caption"),
                "figure_label": meta.get("figure_label"),
                "excluded": False,
                "image_role": meta.get("image_role") or "content",
            }
        )
    out.sort(key=lambda r: (int(r.get("page") or 0), str(r.get("image_path") or "")))
    return out


def split_procedure_blocks(body: str) -> list[str]:
    """Split body into step-ish blocks while preserving order."""
    text = (body or "").strip()
    if not text:
        return []
    # Normalize odd whitespace so Step lines split reliably
    text = text.replace("\u00a0", " ").replace("\t", " ")
    if re.search(r"(?im)^Step\s+\d+", text) or re.search(r"(?m)^\d+[.)]\s+\S", text):
        parts = [p.strip() for p in _STEP_SPLIT_RE.split(text) if p and p.strip()]
        # Keep only parts that are steps or meaningful preamble
        if len(parts) >= 2:
            return parts
    # Split before figure caption lines that stand alone
    fig_split = re.split(r"(?im)(?=^Fig(?:ure)?\.?\s*[\d.]+)", text)
    parts = [p.strip() for p in fig_split if p and p.strip()]
    if len(parts) >= 2:
        return parts
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else [text]


def build_interleaved_page(
    body: str,
    page_images: list[dict[str, Any]],
    *,
    max_images: int,
    append_unused_images: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Rebuild one page: step text with figures inserted after mentioning steps."""
    images = sort_images_for_page(page_images)
    used: set[str] = set()
    placed: list[dict[str, Any]] = []
    blocks = split_procedure_blocks(body)
    out_parts: list[str] = []

    _caption_only = re.compile(
        r"(?is)^\s*(?:Fig(?:ure)?\.?\s*[\d.]+(?:\s*[:\-–].*)?\s*)+$"
    )

    for block in blocks:
        refs = extract_figure_refs(block)
        # Drop standalone caption lines once we can render the real image
        if _caption_only.match(block.strip()) and images:
            for ref in refs:
                if len(placed) >= max_images:
                    break
                img = match_image_for_fig_ref(ref, images, used)
                if not img:
                    continue
                ip = str(img.get("image_path") or "")
                if not ip or ip in used:
                    continue
                used.add(ip)
                placed.append(img)
                out_parts.append(_markdown_image(img))
                out_parts.append("")
            continue

        # Remove trailing standalone Fig. caption lines (rendered as images instead)
        cleaned = re.sub(
            r"(?im)(?:\n|^)\s*Fig(?:ure)?\.?\s*[\d.]+\s*[:\-–].*$",
            "",
            block,
        ).rstrip()
        # Also drop bare "Figure 413" lines that duplicate the image we insert
        cleaned = re.sub(
            r"(?im)(?:\n|^)\s*Fig(?:ure)?\.?\s*[\d.]+\s*$",
            "",
            cleaned,
        ).rstrip()
        if cleaned:
            out_parts.append(cleaned)
        if not refs:
            # Only auto-attach next image when the block explicitly mentions a figure
            if images and re.search(r"(?i)\bfig(?:ure)?\b", block):
                refs = ["__next__"]
        for ref in refs:
            if len(placed) >= max_images:
                break
            if ref == "__next__":
                img = None
                for cand in images:
                    ip = str(cand.get("image_path") or "")
                    if ip and ip not in used:
                        img = cand
                        break
            else:
                img = match_image_for_fig_ref(ref, images, used)
            if not img:
                continue
            ip = str(img.get("image_path") or "")
            if not ip or ip in used:
                continue
            used.add(ip)
            placed.append(img)
            out_parts.append("")
            out_parts.append(_markdown_image(img))
            out_parts.append("")

    if append_unused_images:
        for img in images:
            if len(placed) >= max_images:
                break
            ip = str(img.get("image_path") or "")
            if ip and ip not in used:
                used.add(ip)
                placed.append(img)
                out_parts.append("")
                out_parts.append(_markdown_image(img))
                out_parts.append("")

    return "\n".join(out_parts).strip(), placed


def build_faithful_answer(
    text_hits: list[dict[str, Any]],
    images: list[dict[str, Any]],
    *,
    query: str = "",
) -> dict[str, Any] | None:
    """
    Build a concise document-faithful interleaved answer for the best-matching
    page/section only (avoids dumping neighboring out-of-context pages).
    """
    cfg = load_config()
    if not cfg.get("document_faithful_answers", True):
        return None
    if not text_hits:
        return None

    procedural = [h for h in text_hits if looks_like_procedure(str(h.get("body") or ""))]
    if not procedural:
        if not query_wants_procedure(query):
            return None
        procedural = list(text_hits)

    max_images = int(cfg.get("faithful_max_images", cfg.get("max_images_per_answer", 4)))
    max_pages = int(cfg.get("faithful_max_pages", 1))
    focused = select_focused_hits(procedural, query, max_pages=max_pages)
    if not focused:
        focused = procedural[:1]

    images = overlay_figure_labels_from_okf(images)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for img in images:
        page = int(img.get("page") or 0)
        by_page.setdefault(page, []).append(img)

    seen_bodies: set[str] = set()
    ordered_pages: list[tuple[int, str, dict[str, Any]]] = []
    for h in focused:
        page = int(h.get("page") or 0)
        source = str(h.get("source") or "")
        body = str(h.get("body") or "").strip()
        # Prefer live PDF extraction so bold/italic/underline match the document
        try:
            from app.rich_text import extract_pdf_page_markdown

            rich = extract_pdf_page_markdown(source, page)
            if rich and len(rich.strip()) > 40:
                body = rich
        except Exception as exc:  # noqa: BLE001
            logger.debug("rich extract skipped: %s", exc)
        if not body:
            continue
        body = focus_body_on_query(body, query)
        try:
            from app.rich_text import normalize_list_markers

            body = normalize_list_markers(body)
        except Exception:  # noqa: BLE001
            pass
        key = str(h.get("content_hash") or "") or body[:200]
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        ordered_pages.append((page, body, h))

    # Contiguous continuation (e.g. Practical Activity spanning pp. 25–26)
    cont_pages = int(cfg.get("faithful_continuation_pages", 1))
    if ordered_pages and cont_pages > 0:
        page0, body0, hit0 = ordered_pages[0]
        source0 = str(hit0.get("source") or "")
        have_pages = {p for p, _, _ in ordered_pages}
        if needs_continuation_page(body0, query) and source0:
            for offset in range(1, cont_pages + 1):
                npage = page0 + offset
                if npage in have_pages:
                    continue
                try:
                    from app.rich_text import extract_pdf_page_markdown, normalize_list_markers

                    nrich = extract_pdf_page_markdown(source0, npage)
                except Exception:  # noqa: BLE001
                    nrich = ""
                if not nrich or len(nrich.strip()) < 40:
                    break
                nbody = focus_body_on_query(nrich, query)
                # If focus emptied because keywords are only on prior page, keep
                # continuation when Practical Activity / Step N continues.
                if len((nbody or "").strip()) < 40:
                    if re.search(r"(?i)practical\s+activity|^\s*(?:\*\*)?Step\s+\d+", nrich, re.M):
                        nbody = nrich
                    else:
                        break
                try:
                    nbody = normalize_list_markers(nbody)
                except Exception:  # noqa: BLE001
                    pass
                ordered_pages.append((npage, nbody, hit0))
                have_pages.add(npage)
                for img in load_okf_images_for_pages(source0, [npage]):
                    by_page.setdefault(npage, []).append(img)
                if not needs_continuation_page(nbody, query):
                    break

    if not ordered_pages:
        return None

    if not any(looks_like_procedure(b) for _, b, _ in ordered_pages) and not query_wants_procedure(
        query
    ):
        # Still allow focused factual excerpts that are the best hit
        if score_body_for_query(ordered_pages[0][1], query) < 2.0:
            return None

    sections: list[str] = []
    all_placed: list[dict[str, Any]] = []
    steps_payload: list[dict[str, Any]] = []
    citation_hit = ordered_pages[0][2]

    for page, body, hit in ordered_pages:
        # Only images from this focused page (and matching figure labels when possible)
        page_imgs = by_page.get(page, [])
        wanted = set(extract_figure_refs(query)) | set(extract_figure_refs(body))
        if wanted:
            filtered = []
            for img in page_imgs:
                label = str(img.get("figure_label") or "").strip()
                if label and label in wanted:
                    filtered.append(img)
            # Keep page images if label filter emptied (labels may be missing)
            page_imgs = filtered or page_imgs

        section, placed = build_interleaved_page(
            body,
            page_imgs,
            max_images=max(0, max_images - len(all_placed)),
            append_unused_images=False,
        )
        if not section:
            continue
        # Concise: omit bulky page header when single-page focused answer
        if len(ordered_pages) == 1:
            sections.append(section)
        else:
            header = f"**[{hit.get('source') or ''} page:{page}]**"
            sections.append(f"{header}\n\n{section}")
        all_placed.extend(placed)
        for block in split_procedure_blocks(body):
            steps_payload.append(
                {
                    "page": page,
                    "text": block,
                    "figure_refs": extract_figure_refs(block),
                }
            )

    if not sections:
        return None

    answer = "\n\n---\n\n".join(sections)
    return {
        "answer": answer,
        "images": all_placed,
        "steps": steps_payload,
        "mode": "document_faithful",
        "citation_hit": citation_hit,
    }
