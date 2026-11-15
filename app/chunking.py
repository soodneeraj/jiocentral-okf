"""Structure-aware, page-aware chunking with image association hooks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkDraft:
    page_number: int
    section_heading: str
    parent_section: str
    chunk_type: str
    text: str
    associated_image_ids: list[str] = field(default_factory=list)
    associated_table_ids: list[str] = field(default_factory=list)
    source_coordinates: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def finalize(self) -> ChunkDraft:
        payload = f"{self.page_number}|{self.section_heading}|{self.chunk_type}|{self.text}"
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self


def pack_paragraphs(
    paragraphs: list[str],
    *,
    page_number: int,
    section_heading: str = "",
    parent_section: str = "",
    max_chars: int = 1500,
    overlap_chars: int = 120,
    chunk_type: str = "text",
) -> list[ChunkDraft]:
    """Pack paragraphs into overlapping chunks without splitting mid-paragraph when possible."""
    drafts: list[ChunkDraft] = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) > max_chars and buf:
            drafts.append(
                ChunkDraft(
                    page_number=page_number,
                    section_heading=section_heading,
                    parent_section=parent_section,
                    chunk_type=chunk_type,
                    text=buf,
                ).finalize()
            )
            # overlap
            tail = buf[-overlap_chars:] if overlap_chars and len(buf) > overlap_chars else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
        else:
            buf = candidate
    if buf.strip():
        drafts.append(
            ChunkDraft(
                page_number=page_number,
                section_heading=section_heading,
                parent_section=parent_section,
                chunk_type=chunk_type,
                text=buf.strip(),
            ).finalize()
        )
    return drafts


def chunk_table_markdown(
    md_table: str,
    *,
    page_number: int,
    title: str = "",
    max_chars: int = 3000,
) -> list[ChunkDraft]:
    """Chunk tables while repeating header row in subsequent pieces."""
    lines = [ln for ln in md_table.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0]
    sep = lines[1] if len(lines) > 1 and re.match(r"^\s*\|?\s*[-:]+", lines[1]) else None
    body = lines[2:] if sep else lines[1:]
    prefix = f"{title}\n\n" if title else ""
    header_block = "\n".join([header] + ([sep] if sep else []))

    drafts: list[ChunkDraft] = []
    buf_rows: list[str] = []
    for row in body or [""]:
        trial = prefix + header_block + "\n" + "\n".join(buf_rows + [row])
        if len(trial) > max_chars and buf_rows:
            text = prefix + header_block + "\n" + "\n".join(buf_rows)
            drafts.append(
                ChunkDraft(
                    page_number=page_number,
                    section_heading=title,
                    parent_section=title,
                    chunk_type="table",
                    text=text,
                ).finalize()
            )
            buf_rows = [row]
        else:
            buf_rows.append(row)
    if buf_rows or not body:
        text = prefix + header_block + ("\n" + "\n".join(buf_rows) if buf_rows else "")
        drafts.append(
            ChunkDraft(
                page_number=page_number,
                section_heading=title,
                parent_section=title,
                chunk_type="table",
                text=text,
            ).finalize()
        )
    return drafts


def link_chunks(chunk_ids: list[str]) -> list[tuple[str | None, str | None]]:
    """Return (prev, next) pairs for an ordered list of chunk IDs."""
    links: list[tuple[str | None, str | None]] = []
    for i, _cid in enumerate(chunk_ids):
        prev_id = chunk_ids[i - 1] if i > 0 else None
        next_id = chunk_ids[i + 1] if i + 1 < len(chunk_ids) else None
        links.append((prev_id, next_id))
    return links


def associate_images_to_chunks(
    chunks: list[ChunkDraft],
    page_image_ids: dict[int, list[str]],
) -> list[ChunkDraft]:
    """Attach same-page image occurrence/asset IDs to chunks."""
    for c in chunks:
        c.associated_image_ids = list(page_image_ids.get(c.page_number, []))
    return chunks
