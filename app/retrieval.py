"""Hybrid multimodal retrieval: text + image search, RRF fusion, intent routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.index import embed_text, get_milvus_client
from app.logger import get_logger
from app.utils import load_config

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DIAGRAM = "diagram"
    CROSS_DOC = "cross_document"
    MIXED = "mixed"


@dataclass
class RetrievalCandidate:
    id: str
    modality: str  # text | image
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


TEXT_FIELDS = [
    "type",
    "title",
    "source",
    "source_loc",
    "page",
    "tags",
    "file_path",
    "version",
    "section",
    "body",
    "chunk_id",
    "document_id",
    "associated_image_ids",
    "content_hash",
    "access_scope",
]

IMAGE_FIELDS = [
    "type",
    "title",
    "source",
    "source_loc",
    "page",
    "tags",
    "file_path",
    "image_path",
    "version",
    "section",
    "excluded",
    "caption",
    "image_role",
    "occurrence_id",
    "global_image_asset_id",
    "document_id",
    "associated_chunk_ids",
    "content_hash",
    "ocr_text",
    "image_description",
    "access_scope",
]


def classify_intent(query: str) -> QueryIntent:
    q = query.lower()
    if any(k in q for k in ("compare", "across documents", "both documents", "versus", " vs ")):
        return QueryIntent.CROSS_DOC
    if any(k in q for k in ("diagram", "architecture", "topology", "flowchart", "show me the")):
        return QueryIntent.DIAGRAM
    if any(k in q for k in ("image", "figure", "screenshot", "photo", "picture")):
        return QueryIntent.IMAGE
    if any(k in q for k in ("table", "chart", "trend", "growth", "q1", "q2", "revenue")):
        return QueryIntent.TABLE
    if any(k in q for k in ("diagram", "figure")) and any(k in q for k in ("explain", "what")):
        return QueryIntent.MIXED
    return QueryIntent.TEXT


def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    if not history:
        return query
    # Resolve simple follow-ups
    follow = re.search(
        r"\b(it|that|this|the (?:second|first|previous|above) (?:one|component|diagram|figure))\b",
        query,
        re.I,
    )
    prior_user = ""
    prior_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant" and not prior_assistant:
            prior_assistant = msg.get("content") or ""
        if msg.get("role") == "user" and not prior_user:
            prior_user = msg.get("content") or ""
        if prior_user and prior_assistant:
            break
    if follow and prior_user:
        return f"{prior_user}\nFollow-up: {query}"
    if prior_user and len(query.split()) <= 8:
        return f"{prior_user} {query}"
    return query


def _search_collection(
    collection: str,
    vector: list[float],
    *,
    top_k: int,
    output_fields: list[str],
    filt: str | None = None,
) -> list[dict[str, Any]]:
    client = get_milvus_client()
    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "data": [vector],
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filt:
        kwargs["filter"] = filt
    try:
        results = client.search(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus search failed on %s: %s", collection, exc)
        return []
    rows = results[0] if results else []
    hits: list[dict[str, Any]] = []
    for row in rows:
        entity = row.get("entity") or row
        payload = {k: entity.get(k) for k in output_fields if k in entity}
        for k in output_fields:
            if (k not in payload or payload[k] is None) and k in row:
                payload[k] = row[k]
        payload["distance"] = float(row.get("distance", row.get("score", 0.0)))
        payload["id"] = str(row.get("id") or payload.get("chunk_id") or payload.get("occurrence_id") or "")
        hits.append(payload)
    return hits


def _access_filter(access_scopes: list[str] | None) -> str | None:
    if not access_scopes:
        return None
    # Milvus filter: access_scope in [...]
    quoted = ", ".join(f'"{s}"' for s in access_scopes)
    return f"access_scope in [{quoted}]"


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievalCandidate]],
    *,
    k: int = 60,
) -> list[RetrievalCandidate]:
    scores: dict[str, float] = {}
    merged: dict[str, RetrievalCandidate] = {}
    for list_name, candidates in ranked_lists.items():
        for rank, cand in enumerate(candidates, start=1):
            key = f"{cand.modality}:{cand.id}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in merged:
                merged[key] = cand
            merged[key].ranks[list_name] = rank
            merged[key].score = scores[key]
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 10,
    history: list[dict[str, str]] | None = None,
    access_scopes: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    intent = classify_intent(query) if cfg.get("query_intent_enabled", True) else QueryIntent.MIXED
    search_q = rewrite_query(query, history)
    vector = embed_text(
        f"Represent this sentence for searching relevant passages: {search_q}"
    )
    filt = _access_filter(access_scopes)
    text_top = top_k if intent != QueryIntent.IMAGE else max(3, top_k // 2)
    image_top = top_k if intent in {QueryIntent.IMAGE, QueryIntent.DIAGRAM, QueryIntent.MIXED} else max(3, top_k // 2)

    text_hits = _search_collection(
        cfg["text_collection"],
        vector,
        top_k=text_top,
        output_fields=TEXT_FIELDS,
        filt=filt,
    )
    image_hits: list[dict[str, Any]] = []
    if cfg.get("enable_image_search", True):
        image_hits = _search_collection(
            cfg["image_collection"],
            vector,
            top_k=image_top,
            output_fields=IMAGE_FIELDS,
            filt=filt,
        )
        # Drop excluded / header_footer if present
        image_hits = [
            h
            for h in image_hits
            if h.get("excluded") is not True
            and str(h.get("image_role") or "").lower()
            not in {"header_footer", "header", "footer", "watermark"}
        ]

    text_cands = [
        RetrievalCandidate(
            id=str(h.get("chunk_id") or h.get("file_path") or h.get("id") or i),
            modality="text",
            score=float(h.get("distance") or 0),
            payload=h,
        )
        for i, h in enumerate(text_hits)
    ]
    image_cands = [
        RetrievalCandidate(
            id=str(h.get("occurrence_id") or h.get("image_path") or h.get("id") or i),
            modality="image",
            score=float(h.get("distance") or 0),
            payload=h,
        )
        for i, h in enumerate(image_hits)
    ]

    fused = reciprocal_rank_fusion(
        {"text": text_cands, "image": image_cands},
        k=int(cfg.get("rrf_k", 60)),
    )[: top_k * 2]

    # Page expansion: pull related images for top text hits
    expanded_images = _expand_page_images(text_hits[:3], cfg)

    return {
        "intent": intent.value,
        "search_query": search_q,
        "text_hits": text_hits,
        "image_hits": image_hits,
        "fused": [
            {"id": c.id, "modality": c.modality, "score": c.score, "payload": c.payload}
            for c in fused
        ],
        "expanded_images": expanded_images,
    }


def _expand_page_images(text_hits: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from app.chat import find_related_images

    limit = int(cfg.get("max_images_per_answer", 4))
    expansion = int(cfg.get("page_expansion_limit", 2))
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for h in text_hits:
        source = str(h.get("source") or "")
        page = int(h.get("page") or 0)
        if not source:
            continue
        for delta in range(-expansion, expansion + 1):
            imgs = find_related_images(source, page + delta, limit=5)
            for img in imgs:
                ip = str(img.get("image_path") or "")
                if ip and ip not in seen:
                    seen.add(ip)
                    img["mapped_from_page"] = page
                    collected.append(img)
                if len(collected) >= limit:
                    return collected
    return collected


def heuristic_rerank(text_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(h: dict[str, Any]) -> tuple[int, int, float]:
        is_table = 0 if str(h.get("type") or "") == "table" else 1
        body = str(h.get("body") or "")
        low = 1 if len(body) < 300 and body.endswith("?") else 0
        return (is_table, low, -float(h.get("distance") or 0))

    return sorted(text_hits, key=key)
