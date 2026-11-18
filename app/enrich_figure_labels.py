"""Enrich figure OKF with figure_label from same-page text Fig. refs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from app.faithful import extract_figure_refs
from app.logger import get_logger
from app.okf_spec import validate_okf_file
from app.utils import concepts_dir

logger = get_logger(__name__)

_FIG_FILE_RE = re.compile(r"fig(\d+)", re.I)


def _rewrite(path: Path, meta: dict[str, Any], body: str) -> None:
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{fm}\n---\n\n{body.strip()}\n", encoding="utf-8")


def _fig_index(path: Path, meta: dict[str, Any]) -> int:
    ip = str(meta.get("image_path") or path.name)
    m = _FIG_FILE_RE.search(ip)
    return int(m.group(1)) if m else 9999


def enrich_figure_labels(doc_filter: str | None = None) -> dict[str, int]:
    """
    For each (source, page), collect ordered Fig. refs from text bodies and
    assign them to figure concepts in figN order when counts allow.
    """
    root = concepts_dir()
    if not root.exists():
        return {"docs": 0, "updated": 0, "skipped": 0}

    text_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    figs_by_key: dict[tuple[str, int], list[tuple[Path, dict[str, Any], str]]] = defaultdict(
        list
    )

    for md_path in root.rglob("*.md"):
        if doc_filter and doc_filter.lower() not in str(md_path).lower():
            continue
        try:
            meta, body = validate_okf_file(md_path)
        except ValueError:
            continue
        ctype = meta.get("type")
        source = str(meta.get("source") or "")
        page = int(meta.get("page") or 0)
        if not page:
            loc = str(meta.get("source_loc") or "")
            if "page:" in loc:
                try:
                    page = int(loc.split("page:", 1)[1].split()[0])
                except ValueError:
                    page = 0
        if not source or page <= 0:
            continue
        key = (source, page)
        if ctype == "document":
            text_by_key[key].append(body)
        elif ctype == "figure" and meta.get("excluded") is not True:
            figs_by_key[key].append((md_path, meta, body))

    stats = {"docs": 0, "updated": 0, "skipped": 0, "pages": 0}
    seen_sources: set[str] = set()

    for key, fig_rows in figs_by_key.items():
        source, page = key
        seen_sources.add(source)
        stats["pages"] += 1
        refs: list[str] = []
        seen_refs: set[str] = set()
        for body in text_by_key.get(key, []):
            for ref in extract_figure_refs(body):
                if ref not in seen_refs:
                    seen_refs.add(ref)
                    refs.append(ref)

        fig_rows_sorted = sorted(fig_rows, key=lambda row: _fig_index(row[0], row[1]))
        if not refs:
            stats["skipped"] += len(fig_rows_sorted)
            continue

        # Zip refs onto figures in order (textbook layout)
        n = min(len(refs), len(fig_rows_sorted))
        for i in range(n):
            path, meta, body = fig_rows_sorted[i]
            label = refs[i]
            caption = str(meta.get("caption") or "").strip()
            new_caption = f"Fig. {label}"
            # Keep richer captions if already labeled
            if caption.lower().startswith("fig.") and label in caption:
                new_caption = caption
            elif caption and not caption.lower().startswith("figure from"):
                if label not in caption:
                    new_caption = f"Fig. {label} — {caption}"
                else:
                    new_caption = caption
            meta["figure_label"] = label
            meta["caption"] = new_caption
            # Mirror into body caption line when present
            lines = body.splitlines()
            rewritten = False
            for li, line in enumerate(lines):
                if line.lower().startswith("caption:"):
                    lines[li] = f"Caption: {new_caption}"
                    rewritten = True
                    break
            if not rewritten:
                lines.append("")
                lines.append(f"Caption: {new_caption}")
            _rewrite(path, meta, "\n".join(lines))
            stats["updated"] += 1
            logger.info(
                "Labeled %s → Fig. %s (source=%s page=%d)",
                path.name,
                label,
                source,
                page,
            )

        stats["skipped"] += max(0, len(fig_rows_sorted) - n)

    stats["docs"] = len(seen_sources)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich figure OKF with Fig. labels")
    parser.add_argument(
        "--doc",
        default="",
        help="Optional substring filter (e.g. jhde101)",
    )
    parser.add_argument(
        "--reindex-images",
        action="store_true",
        help="After enrichment, recreate+reindex the image Milvus collection only",
    )
    args = parser.parse_args()
    stats = enrich_figure_labels(doc_filter=args.doc or None)
    print(f"enrich_figure_labels: {stats}")
    if args.reindex_images:
        from app.index import index_okf

        idx = index_okf(recreate=False, images_only=True, recreate_images=True)
        print(f"image reindex: {idx}")


if __name__ == "__main__":
    main()
