"""Preserve PDF text formatting (bold / italic / underline / headings) as Markdown."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.logger import get_logger
from app.utils import source_pdfs_dir

logger = get_logger(__name__)

# PyMuPDF span flags
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4


def _is_bold(span: dict[str, Any]) -> bool:
    flags = int(span.get("flags") or 0)
    font = str(span.get("font") or "")
    if flags & _FLAG_BOLD:
        return True
    return bool(re.search(r"(?i)bold|black|heavy|demi", font))


def _is_italic(span: dict[str, Any]) -> bool:
    flags = int(span.get("flags") or 0)
    font = str(span.get("font") or "")
    if flags & _FLAG_ITALIC:
        return True
    return bool(re.search(r"(?i)italic|oblique", font))


def _wrap_style(text: str, *, bold: bool, italic: bool, underline: bool) -> str:
    t = text
    if not t or not t.strip():
        return t
    # Preserve leading/trailing whitespace outside markers
    lead = re.match(r"^\s*", t).group(0)  # type: ignore[union-attr]
    trail = re.search(r"\s*$", t).group(0)  # type: ignore[union-attr]
    core = t[len(lead) : len(t) - len(trail) if trail else len(t)]
    if not core:
        return t
    if underline:
        core = f"<u>{core}</u>"
    if bold and italic:
        core = f"***{core}***"
    elif bold:
        core = f"**{core}**"
    elif italic:
        core = f"*{core}*"
    return f"{lead}{core}{trail}"


def _span_underlined(
    span: dict[str, Any],
    line_bbox: tuple[float, float, float, float] | None,
    underlines: list[tuple[float, float, float, float]],
) -> bool:
    """Heuristic: a near-horizontal drawing sits just under the span."""
    sb = span.get("bbox")
    if not sb or not underlines:
        return False
    x0, y0, x1, y1 = sb
    for ux0, uy0, ux1, uy1 in underlines:
        # line roughly under the span baseline
        if abs(uy0 - y1) <= 3.5 or abs(uy1 - y1) <= 3.5:
            # horizontal overlap
            overlap = min(x1, ux1) - max(x0, ux0)
            if overlap > 0.4 * max(1.0, x1 - x0):
                return True
    return False


def _collect_underlines(page: Any) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    try:
        for d in page.get_drawings() or []:
            for item in d.get("items") or []:
                if not item:
                    continue
                op = item[0]
                if op == "l" and len(item) >= 3:
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) <= 1.5:  # horizontal
                        x0, x1 = sorted([p1.x, p2.x])
                        out.append((x0, p1.y, x1, p2.y))
                elif op == "re" and len(item) >= 2:
                    r = item[1]
                    # very flat rectangle ≈ underline
                    if getattr(r, "height", 99) <= 2.0 and getattr(r, "width", 0) > 8:
                        out.append((r.x0, r.y1, r.x1, r.y1))
    except Exception:  # noqa: BLE001
        return out
    return out


def _merge_styled_runs(
    runs: list[tuple[str, bool, bool, bool]],
) -> list[tuple[str, bool, bool, bool]]:
    """Merge adjacent spans that share the same bold/italic/underline style."""
    if not runs:
        return []
    merged: list[tuple[str, bool, bool, bool]] = [runs[0]]
    for text, bold, italic, under in runs[1:]:
        pt, pb, pi, pu = merged[-1]
        if (bold, italic, under) == (pb, pi, pu):
            merged[-1] = (pt + text, pb, pi, pu)
        else:
            merged.append((text, bold, italic, under))
    return merged


def page_dict_to_markdown(page: Any, *, detect_underline: bool = False) -> str:
    """Convert a PyMuPDF page to Markdown preserving bold/italic/underline/headings."""
    data = page.get_text("dict") or {}
    # Drawing-based underline is optional — many manuals false-positive on rules/borders
    underlines = _collect_underlines(page) if detect_underline else []
    sizes: list[float] = []
    for block in data.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                t = span.get("text") or ""
                if t.strip():
                    sizes.append(float(span.get("size") or 0))
    median = sorted(sizes)[len(sizes) // 2] if sizes else 11.0
    heading_cut = median * 1.35

    lines_out: list[str] = []
    for block in data.get("blocks") or []:
        if block.get("type") != 0:
            continue
        block_parts: list[str] = []
        for line in block.get("lines") or []:
            line_bbox = tuple(line.get("bbox") or ())  # type: ignore[assignment]
            lb = (
                (line_bbox[0], line_bbox[1], line_bbox[2], line_bbox[3])
                if len(line_bbox) == 4
                else None
            )
            runs: list[tuple[str, bool, bool, bool]] = []
            for span in line.get("spans") or []:
                raw = str(span.get("text") or "")
                if not raw:
                    continue
                size = float(span.get("size") or 0)
                bold = _is_bold(span)
                italic = _is_italic(span)
                under = (
                    _span_underlined(span, lb, underlines) if detect_underline else False
                )
                # Font-name underline hints (rare)
                font = str(span.get("font") or "")
                if re.search(r"(?i)underline", font):
                    under = True
                if size >= heading_cut and raw.strip():
                    bold = True
                runs.append((raw, bold, italic, under))
            pieces = [
                _wrap_style(t, bold=b, italic=i, underline=u)
                for t, b, i, u in _merge_styled_runs(runs)
            ]
            line_text = "".join(pieces).rstrip()
            if not line_text.strip():
                continue
            block_parts.append(line_text)
        if block_parts:
            lines_out.append("\n".join(block_parts))
    return "\n\n".join(lines_out).strip()


def resolve_source_pdf(source: str) -> Path | None:
    """Find a local PDF for a Milvus/OKF source filename."""
    name = Path(str(source or "")).name
    if not name:
        return None
    root = source_pdfs_dir()
    direct = root / name
    if direct.is_file():
        return direct
    # Fuzzy: stem match ignoring trailing timestamps
    stem = Path(name).stem
    for p in root.glob("*.pdf"):
        if p.name == name or p.stem == stem or p.stem.startswith(stem):
            return p
    return None


def extract_pdf_page_markdown(source: str, page: int) -> str:
    """Extract one PDF page as formatted Markdown. Never raises."""
    if page <= 0:
        return ""
    path = resolve_source_pdf(source)
    if path is None:
        return ""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        try:
            if page > len(doc):
                return ""
            return page_dict_to_markdown(doc[page - 1])
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("rich text extract failed for %s p%s: %s", source, page, exc)
        return ""


# Wingdings/PDF tick boxes and common check/box glyphs used as list markers
_BOX_BULLET_CHARS = (
    "\uf0fc"  #  (common PDF Wingdings check)
    "\u2713"  # ✓
    "\u2714"  # ✔
    "\u2611"  # ☑
    "\u2610"  # ☐
    "\u25a1"  # □
    "\u25a0"  # ■
    "\u25a2"  # ▢
    "\u274f"  # ❏
    "\u2751"  # ❑
    "\u2752"  # ❒
)
_BOX_BULLET_RE = re.compile(
    rf"(?m)^(\s*)[{re.escape(_BOX_BULLET_CHARS)}]\s*"
)


def normalize_list_markers(text: str) -> str:
    """Convert checkbox/tick list markers ( etc.) to solid round bullets (•)."""
    if not text:
        return text
    return _BOX_BULLET_RE.sub(r"\1• ", text)


def markdown_inline_to_html(text: str) -> str:
    """
    Convert a limited Markdown subset to HTML for Document view.

    Supports: **bold**, ***bold italic***, *italic*, <u>underline</u>,
    and preserves newlines as <br/>.
    """
    import html as _html

    s = text or ""
    # Protect existing <u>...</u> then escape, then restore + apply md
    u_slots: list[str] = []

    def _park_u(m: re.Match[str]) -> str:
        u_slots.append(m.group(1))
        return f"\x00U{len(u_slots) - 1}\x00"

    s = re.sub(r"<u>(.*?)</u>", _park_u, s, flags=re.I | re.S)
    s = _html.escape(s)
    # Restore underline placeholders as tags around escaped content
    for i, inner in enumerate(u_slots):
        s = s.replace(f"\x00U{i}\x00", f"<u>{_html.escape(inner)}</u>")

    # Bold+italic, bold, italic (non-greedy, no cross-newline for italic single *)
    # Apply markdown markers; allow tags already present inside (e.g. <u>)
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", s, flags=re.S)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s, flags=re.S)
    s = re.sub(
        r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)",
        r"<em>\1</em>",
        s,
    )
    s = re.sub(r"__(.+?)__", r"<u>\1</u>", s, flags=re.S)
    # Collapse accidental nested empties from bad merges
    s = re.sub(r"</strong>\s*<strong>", "", s)
    s = re.sub(r"</em>\s*<em>", "", s)
    s = re.sub(r"</u>\s*<u>", "", s)
    s = s.replace("\n", "<br/>")
    return s
