"""OKF bundle → Milvus: embed text/figures and caption images via Ollama."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import yaml
from pymilvus import MilvusClient

from app.logger import get_logger
from app.okf_spec import validate_okf_file
from app.utils import concepts_dir, load_config, okf_bundle_dir

logger = get_logger(__name__)

_PAGE_RE = re.compile(r"page:(\d+)", re.IGNORECASE)

_milvus_client: MilvusClient | None = None


def parse_page_number(source_loc: str) -> int:
    """Parse page number from ``source_loc`` (``page:N``); return 0 if unparseable."""
    if not source_loc:
        return 0
    m = _PAGE_RE.search(str(source_loc))
    return int(m.group(1)) if m else 0


def get_milvus_client() -> MilvusClient:
    """Module-level singleton MilvusClient."""
    global _milvus_client
    if _milvus_client is None:
        cfg = load_config()
        kwargs: dict[str, Any] = {"uri": cfg["milvus_uri"]}
        user = cfg.get("milvus_user") or ""
        password = cfg.get("milvus_password") or ""
        if user:
            kwargs["token"] = f"{user}:{password}"
        _milvus_client = MilvusClient(**kwargs)
    return _milvus_client


def _l2_normalize(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def embed_text(text: str, retries: int = 3) -> list[float]:
    """Embed via Ollama embedding model; always L2-normalize the result."""
    cfg = load_config()
    client = ollama.Client(host=cfg["ollama_url"])
    # nomic-embed-text context is short (~2048 tokens); keep prompts bounded
    prompt = (text or "")[:3000]
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.embeddings(model=cfg["embedding_model"], prompt=prompt)
            if isinstance(resp, dict):
                vector = resp.get("embedding") or resp.get("embeddings")
            else:
                vector = getattr(resp, "embedding", None) or getattr(
                    resp, "embeddings", None
                )
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]
            if not vector:
                raise RuntimeError(f"Empty embedding response: {resp!r}")
            return _l2_normalize(list(vector))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2**attempt
            logger.warning(
                "embed_text attempt %d/%d failed: %s — retrying in %ds",
                attempt + 1,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def caption_image_with_llava(image_path: Path) -> str:
    """Caption a figure via Ollama vision model. Never raises."""
    cfg = load_config()
    prompt = (
        "Describe this enterprise figure/diagram/chart in 2 detailed "
        "sentences. Mention numbers, entities, and trends where visible."
    )
    try:
        client = ollama.Client(host=cfg["ollama_url"])
        resp = client.generate(
            model=cfg["vision_model"],
            prompt=prompt,
            images=[str(image_path)],
        )
        if isinstance(resp, dict):
            caption = (resp.get("response") or "").strip()
        else:
            caption = (getattr(resp, "response", None) or "").strip()
        if not caption:
            caption = f"Figure from {image_path.name}"
        return caption
    except Exception as exc:  # noqa: BLE001
        logger.warning("caption_image_with_llava failed for %s: %s", image_path, exc)
        return f"Figure from {image_path.name}"


def _ensure_collection(
    client: MilvusClient,
    name: str,
    dimension: int,
    metric_type: str,
    recreate: bool,
) -> None:
    exists = client.has_collection(name)
    if exists and recreate:
        logger.info("Dropping collection %s (recreate=True)", name)
        client.drop_collection(name)
        exists = False
    if not exists:
        logger.info(
            "Creating collection %s (dim=%d, metric=%s, dynamic fields)",
            name,
            dimension,
            metric_type,
        )
        client.create_collection(
            collection_name=name,
            dimension=dimension,
            metric_type=metric_type,
            auto_id=True,
            enable_dynamic_field=True,
        )


def delete_by_document_id(document_id: str) -> dict[str, int]:
    """Remove all Milvus rows for a document from text and image collections."""
    cfg = load_config()
    client = get_milvus_client()
    deleted = {"text": 0, "image": 0}
    filt = f'document_id == "{document_id}"'
    for key, col in (("text", cfg["text_collection"]), ("image", cfg["image_collection"])):
        try:
            if not client.has_collection(col):
                continue
            result = client.delete(collection_name=col, filter=filt)
            # pymilvus may return dict with delete_count
            if isinstance(result, dict):
                deleted[key] = int(result.get("delete_count") or result.get("count") or 0)
            else:
                deleted[key] = int(getattr(result, "delete_count", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_by_document_id failed on %s: %s", col, exc)
    return deleted


def delete_by_source(source: str) -> dict[str, int]:
    """Legacy cleanup by PDF filename (source field)."""
    cfg = load_config()
    client = get_milvus_client()
    deleted = {"text": 0, "image": 0}
    safe = source.replace('"', '\\"')
    filt = f'source == "{safe}"'
    for key, col in (("text", cfg["text_collection"]), ("image", cfg["image_collection"])):
        try:
            if not client.has_collection(col):
                continue
            result = client.delete(collection_name=col, filter=filt)
            if isinstance(result, dict):
                deleted[key] = int(result.get("delete_count") or result.get("count") or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_by_source failed on %s: %s", col, exc)
    return deleted


def _iter_concept_files() -> list[Path]:
    """Yield concept files with text/table docs before figures.

    Alphabetical sort puts ``figure_*.md`` ahead of ``text_*.md``, which delays
    populating ``jcrag_text_chunks`` (the only collection chat searches) for a
    long time on image-heavy corpora. Prefer retrieval-critical types first.
    """
    root = concepts_dir()
    if not root.exists():
        return []

    def _priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if name.startswith("figure_"):
            return (1, str(path).lower())
        return (0, str(path).lower())

    return sorted(root.rglob("*.md"), key=_priority)


def _rewrite_file(path: Path, meta: dict[str, Any], body: str) -> None:
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{fm}\n---\n\n{body.strip()}\n", encoding="utf-8")


def index_okf(
    recreate: bool = False,
    *,
    images_only: bool = False,
    recreate_images: bool = False,
) -> dict[str, int]:
    """Embed OKF concepts into Milvus. Optionally reindex images only."""
    cfg = load_config()
    client = get_milvus_client()
    dim = int(cfg["embedding_dim"])
    metric = cfg.get("milvus_metric", "IP")
    text_col = cfg["text_collection"]
    image_col = cfg["image_collection"]

    if not images_only:
        _ensure_collection(client, text_col, dim, metric, recreate)
    _ensure_collection(
        client,
        image_col,
        dim,
        metric,
        recreate or recreate_images,
    )

    text_batch: list[dict[str, Any]] = []
    image_batch: list[dict[str, Any]] = []
    stats = {"text": 0, "image": 0, "skipped": 0, "errors": 0}

    def flush(force: bool = False) -> None:
        nonlocal text_batch, image_batch
        if text_batch and (force or len(text_batch) >= 100):
            client.insert(collection_name=text_col, data=text_batch)
            logger.info("Flushed %d text rows", len(text_batch))
            text_batch = []
        if image_batch and (force or len(image_batch) >= 100):
            client.insert(collection_name=image_col, data=image_batch)
            logger.info("Flushed %d image rows", len(image_batch))
            image_batch = []

    bundle = okf_bundle_dir()
    for md_path in _iter_concept_files():
        try:
            meta, body = validate_okf_file(md_path)
        except ValueError as exc:
            logger.warning("Skipping invalid OKF file %s: %s", md_path, exc)
            stats["errors"] += 1
            continue

        ctype = meta.get("type")
        if ctype == "index":
            stats["skipped"] += 1
            continue

        # Excluded figures: skip before captioning
        if ctype == "figure" and meta.get("excluded") is True:
            logger.info("Skipping excluded figure %s", md_path)
            stats["skipped"] += 1
            continue

        if images_only and ctype != "figure":
            continue

        title = str(meta.get("title") or "")
        source = str(meta.get("source") or "")
        source_loc = str(meta.get("source_loc") or "")
        tags = meta.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        page = parse_page_number(source_loc)
        if not page and meta.get("page") is not None:
            try:
                page = int(meta.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
        rel_path = str(md_path.relative_to(bundle)).replace("\\", "/")

        caption = ""
        image_path_rel = str(meta.get("image_path") or "")

        if ctype == "figure":
            caption = str(meta.get("caption") or "")
            if not caption:
                for line in body.splitlines():
                    if line.lower().startswith("caption:"):
                        caption = line.split(":", 1)[1].strip()
                        break
            if not caption or caption == "[NEEDS_CAPTION]":
                caption = f"Figure from {Path(image_path_rel).name}" if image_path_rel else title

            # Optional vision captioning (disabled by default — very slow on large corpora)
            if (
                cfg.get("caption_images", False)
                and "[NEEDS_CAPTION]" in body
                and image_path_rel
            ):
                abs_img = bundle / image_path_rel
                if abs_img.exists():
                    caption = caption_image_with_llava(abs_img)
                    body = body.replace("[NEEDS_CAPTION]", caption)
                    meta["caption"] = caption
                    _rewrite_file(md_path, meta, body)
                    logger.info("Captioned %s", md_path.name)
            elif "[NEEDS_CAPTION]" in body:
                body = body.replace("[NEEDS_CAPTION]", caption)
                meta["caption"] = caption
                _rewrite_file(md_path, meta, body)

            # Prefer caption + description + OCR for searchable image text space
            desc = str(meta.get("image_description") or "")
            ocr_txt = str(meta.get("ocr_text") or "")
            figure_label = str(meta.get("figure_label") or "").strip()
            embed_input = f"{title} {caption} Fig. {figure_label} {desc} {ocr_txt}".strip()
            try:
                vector = embed_text(embed_input)
            except Exception as exc:  # noqa: BLE001
                logger.error("Embed failed for figure %s: %s", md_path, exc)
                stats["errors"] += 1
                continue

            row = {
                "vector": vector,
                "type": ctype,
                "title": title,
                "source": source,
                "source_loc": source_loc,
                "page": page,
                "tags": tags,
                "file_path": rel_path,
                "image_path": image_path_rel,
                "caption": caption,
                "figure_label": figure_label,
                "image_description": desc,
                "ocr_text": ocr_txt,
                "excluded": bool(meta.get("excluded") is True),
                "image_role": str(meta.get("image_role") or "content"),
                "occurrence_id": str(meta.get("occurrence_id") or ""),
                "global_image_asset_id": str(meta.get("global_image_asset_id") or ""),
                "document_id": str(meta.get("document_id") or ""),
                "content_hash": str(meta.get("content_hash") or ""),
                "associated_chunk_ids": meta.get("related") or [],
                "embedding_model_version": str(cfg.get("embedding_model") or ""),
                "access_scope": str(meta.get("confidentiality") or cfg.get("default_access_scope") or "internal"),
            }
            if meta.get("version"):
                row["version"] = str(meta["version"])
            if meta.get("section"):
                row["section"] = str(meta["section"])
            if row["excluded"]:
                stats["skipped"] += 1
                continue
            image_batch.append(row)
            stats["image"] += 1
        else:
            embed_input = f"{title} {body[:6000]}".strip()
            try:
                vector = embed_text(embed_input)
            except Exception as exc:  # noqa: BLE001
                logger.error("Embed failed for %s: %s", md_path, exc)
                stats["errors"] += 1
                continue

            row = {
                "vector": vector,
                "type": ctype,
                "title": title,
                "source": source,
                "source_loc": source_loc,
                "page": page,
                "tags": tags,
                "file_path": rel_path,
                "body": body[:12000],
                "chunk_id": str(meta.get("chunk_id") or ""),
                "document_id": str(meta.get("document_id") or ""),
                "content_hash": str(meta.get("content_hash") or ""),
                "associated_image_ids": meta.get("related") or [],
                "embedding_model_version": str(cfg.get("embedding_model") or ""),
                "access_scope": str(meta.get("confidentiality") or cfg.get("default_access_scope") or "internal"),
            }
            if meta.get("version"):
                row["version"] = str(meta["version"])
            if meta.get("section"):
                row["section"] = str(meta["section"])
            text_batch.append(row)
            stats["text"] += 1

        flush(force=False)
        done = stats["text"] + stats["image"] + stats["skipped"] + stats["errors"]
        if done and done % 50 == 0:
            logger.info(
                "Index progress: text=%d image=%d skipped=%d errors=%d",
                stats["text"],
                stats["image"],
                stats["skipped"],
                stats["errors"],
            )

    flush(force=True)
    logger.info(
        "Indexing complete: text=%d image=%d skipped=%d errors=%d",
        stats["text"],
        stats["image"],
        stats["skipped"],
        stats["errors"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed OKF bundle concepts into Milvus."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate Milvus collections before indexing.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only (re)index figure concepts into the image collection.",
    )
    parser.add_argument(
        "--recreate-images",
        action="store_true",
        help="Drop and recreate the image collection (use with --images-only).",
    )
    args = parser.parse_args()
    index_okf(
        recreate=args.recreate,
        images_only=args.images_only,
        recreate_images=args.recreate_images,
    )


if __name__ == "__main__":
    main()
