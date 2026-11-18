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
            # still place matching images if not used
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
        out_parts.append(cleaned if cleaned else block.rstrip())
        if not refs:
            # If block mentions "fig" generically or is a step, attach next unused image
            if images and re.search(r"(?i)\bfig(?:ure)?\b|\bstep\s+\d+", block):
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

    # Append any remaining same-page images not yet placed (cap)
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
    Build a document-faithful interleaved answer when procedure evidence exists.

    Returns None when the path should fall back to normal LLM answering.
    """
    cfg = load_config()
    if not cfg.get("document_faithful_answers", True):
        return None
    if not text_hits:
        return None

    # Prefer hits that look like procedures
    procedural = [h for h in text_hits if looks_like_procedure(str(h.get("body") or ""))]
    if not procedural:
        if not query_wants_procedure(query):
            return None
        procedural = list(text_hits)

    max_images = int(cfg.get("faithful_max_images", cfg.get("max_images_per_answer", 12)))
    images = overlay_figure_labels_from_okf(images)
    # Group images by page
    by_page: dict[int, list[dict[str, Any]]] = {}
    for img in images:
        page = int(img.get("page") or 0)
        by_page.setdefault(page, []).append(img)

    # Deduplicate text bodies by content_hash / body while preserving page order
    seen_bodies: set[str] = set()
    ordered_pages: list[tuple[int, str, dict[str, Any]]] = []
    for h in procedural:
        body = str(h.get("body") or "").strip()
        if not body:
            continue
        key = str(h.get("content_hash") or "") or body[:200]
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        page = int(h.get("page") or 0)
        ordered_pages.append((page, body, h))

    if not ordered_pages:
        return None

    # Only reconstruct when at least one page is procedural or query asks for steps
    if not any(looks_like_procedure(b) for _, b, _ in ordered_pages) and not query_wants_procedure(
        query
    ):
        return None

    sections: list[str] = []
    all_placed: list[dict[str, Any]] = []
    steps_payload: list[dict[str, Any]] = []
    citation_hit = ordered_pages[0][2]

    for page, body, hit in ordered_pages:
        page_imgs = by_page.get(page, [])
        section, placed = build_interleaved_page(
            body, page_imgs, max_images=max(0, max_images - len(all_placed))
        )
        if not section:
            continue
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
