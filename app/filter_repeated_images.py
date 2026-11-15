"""Mark header/footer / repetitive figures as non-indexable in the OKF bundle.

Policy (aligned with master prompt):
- Cross-page repeated hashes → image_role=header_footer, excluded from indexing.
- Exact duplicates keep occurrence concepts; only mark excluded when configured.
- Prefer NOT deleting files so occurrence traceability is preserved.
- Asset registry in ``app.assets`` is authoritative for binary dedupe.

Takes effect in Milvus after ``python -m app.index --recreate`` or incremental reindex.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from app.ingest import _dhash, _render_okf_content
from app.logger import get_logger
from app.okf_spec import validate_okf_file
from app.utils import concepts_dir, load_config, okf_bundle_dir

logger = get_logger(__name__)


def filter_repeated_images(
    min_pages: int | None = None,
    *,
    drop_all_duplicates: bool = False,
    delete_files: bool | None = None,
) -> dict[str, int]:
    """Exclude (and optionally delete) duplicate / cross-page repeated figures."""
    cfg = load_config()
    threshold = int(min_pages or cfg.get("header_footer_min_pages", 2))
    if delete_files is None:
        # Default false — preserve occurrence artifacts for traceability
        delete_files = bool(cfg.get("delete_duplicate_image_files", False))
    bundle = okf_bundle_dir()

    # source -> dhash -> list[(md_path, page, meta, body, abs_img)]
    by_source: dict[str, dict[int, list[tuple]]] = defaultdict(lambda: defaultdict(list))

    for md_path in sorted(concepts_dir().rglob("figure_*.md")):
        try:
            meta, body = validate_okf_file(md_path)
        except ValueError:
            continue
        if meta.get("type") != "figure":
            continue
        ip = str(meta.get("image_path") or "")
        if not ip:
            continue
        abs_img = bundle / ip
        if not abs_img.exists():
            # Concept without file — still treat as removable duplicate candidate
            h = hash(ip)
            source = str(meta.get("source") or "")
            page = int(meta.get("page") or 0)
            by_source[source][h].append((md_path, page, meta, body, abs_img))
            continue
        try:
            img = Image.open(abs_img)
            h = _dhash(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hash failed for %s: %s", abs_img, exc)
            continue
        source = str(meta.get("source") or "")
        page = int(meta.get("page") or 0)
        if not page:
            loc = str(meta.get("source_loc") or "")
            if "page:" in loc:
                try:
                    page = int(loc.split("page:", 1)[1].split()[0])
                except ValueError:
                    page = 0
        by_source[source][h].append((md_path, page, meta, body, abs_img))

    marked = 0
    deleted = 0
    scanned = 0

    for source, hashes in by_source.items():
        for h, entries in hashes.items():
            pages = {p for _, p, _, _, _ in entries}
            scanned += len(entries)
            is_cross_page = len(pages) >= threshold
            is_duplicate = drop_all_duplicates and len(entries) > 1
            if not is_cross_page and not is_duplicate:
                continue

            role = "header_footer" if is_cross_page else "duplicate"
            for md_path, _page, meta, body, abs_img in entries:
                if delete_files:
                    try:
                        if abs_img.exists():
                            abs_img.unlink()
                            deleted += 1
                    except OSError as exc:
                        logger.warning("Failed deleting %s: %s", abs_img, exc)
                    try:
                        md_path.unlink()
                        marked += 1
                    except OSError as exc:
                        logger.warning("Failed deleting concept %s: %s", md_path, exc)
                else:
                    meta["excluded"] = True
                    meta["image_role"] = role
                    tags = list(meta.get("tags") or [])
                    if role not in tags:
                        tags.append(role)
                    if "duplicate" not in tags and role == "duplicate":
                        tags.append("duplicate")
                    meta["tags"] = tags
                    md_path.write_text(_render_okf_content(meta, body), encoding="utf-8")
                    marked += 1

            logger.info(
                "Removed %d %s figures in %s (occurrences=%d pages=%d)",
                len(entries),
                role,
                source,
                len(entries),
                len(pages),
            )

    stats = {
        "scanned": scanned,
        "removed_or_excluded": marked,
        "files_deleted": deleted,
        "min_pages": threshold,
        "drop_all_duplicates": drop_all_duplicates,
        "delete_files": delete_files,
    }
    logger.info("Duplicate image cleanup done: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate images (and cross-page headers/footers). "
            "By default deletes concept+PNG files so they are never loaded. "
            "Re-run `python -m app.index --recreate` afterwards."
        )
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=None,
        help="Min distinct pages for cross-page header/footer removal (default from config)",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Mark excluded in frontmatter instead of deleting files",
    )
    args = parser.parse_args()
    filter_repeated_images(
        min_pages=args.min_pages,
        drop_all_duplicates=True,
        delete_files=not args.keep_files,
    )


if __name__ == "__main__":
    main()
