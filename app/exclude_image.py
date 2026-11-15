"""Admin CLI: mark a figure concept as excluded after human review.

Takes effect only after the next ``python -m app.index --recreate``.
Does not touch Milvus directly or re-caption anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.ingest import _render_okf_content
from app.logger import get_logger
from app.okf_spec import validate_okf_file
from app.utils import concepts_dir

logger = get_logger(__name__)


def _normalize_image_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _find_figure_by_image_path(image_path: str) -> Path | None:
    """Linear scan of every concept file for a matching figure image_path."""
    target = _normalize_image_path(image_path)
    # Also accept bare filenames by prefixing assets/images/
    candidates = {target}
    if not target.startswith("assets/"):
        candidates.add(f"assets/images/{Path(target).name}")
        candidates.add(_normalize_image_path(str(Path("assets/images") / Path(target).name)))

    for md_path in sorted(concepts_dir().rglob("*.md")):
        try:
            meta, _body = validate_okf_file(md_path)
        except ValueError:
            continue
        if meta.get("type") != "figure":
            continue
        ip = _normalize_image_path(str(meta.get("image_path", "")))
        if ip in candidates or Path(ip).name == Path(target).name:
            return md_path
    return None


def exclude_image(image_path: str) -> Path:
    """Set ``excluded: true`` on the matching figure concept and rewrite it."""
    md_path = _find_figure_by_image_path(image_path)
    if md_path is None:
        raise ValueError(f"No figure concept found for image_path={image_path!r}")

    meta, body = validate_okf_file(md_path)
    meta["excluded"] = True
    md_path.write_text(_render_okf_content(meta, body), encoding="utf-8")
    logger.info("Marked excluded=true on %s (image_path=%s)", md_path, image_path)
    logger.info(
        "Re-run `python -m app.index --recreate` for this exclusion to take effect in Milvus."
    )
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mark a figure concept excluded after human review. "
            "Does NOT update Milvus — run `python -m app.index --recreate` afterwards."
        )
    )
    parser.add_argument(
        "image_path",
        help='Path like "assets/images/<doc>_p3_fig0.png"',
    )
    args = parser.parse_args()
    try:
        exclude_image(args.image_path)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
