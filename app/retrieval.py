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
    "figure_label",
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


def normalize_search_query(query: str) -> str:
    """Strip textbook pedagogy so dense search hits narrative facts, not the exercise stem."""
    q = (query or "").strip()
    if not q:
        return q
    q = re.sub(
        r"(?i)^in the story(?:\s+we are told that)?\s*",
        "",
        q,
    )
    q = re.sub(
        r"(?i)\bdiscuss(?:\s+it)?\s+in\s+(?:a\s+)?group\b[.,;:]?\s*",
        "",
        q,
    )
    q = re.sub(
        r"(?i)\b(?:answer the following|write a short note on|explain with reference "
        r"to the context|complete the following)\b[.,;:]?\s*",
        "",
        q,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q or query.strip()


def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    q = normalize_search_query(query)
    if not history:
        return q
    # Resolve simple follow-ups
    follow = re.search(
        r"\b(it|that|this|the (?:second|first|previous|above) (?:one|component|diagram|figure))\b",
        q,
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
        return f"{normalize_search_query(prior_user)}\nFollow-up: {q}"
    if prior_user and len(q.split()) <= 8:
        return f"{normalize_search_query(prior_user)} {q}"
    return q


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
    # Over-fetch text so lexical/heuristic rerank can recover factual needles
    # that dense retrieval buries under exercise/worksheet pages.
    fetch_mult = max(2, int(cfg.get("retrieval_overfetch_factor", 4)))
    text_top = (top_k if intent != QueryIntent.IMAGE else max(3, top_k // 2)) * fetch_mult
    image_top = top_k if intent in {QueryIntent.IMAGE, QueryIntent.DIAGRAM, QueryIntent.MIXED} else max(3, top_k // 2)

    text_hits = _search_collection(
        cfg["text_collection"],
        vector,
        top_k=text_top,
        output_fields=TEXT_FIELDS,
        filt=filt,
    )
    text_hits = heuristic_rerank(text_hits, query=search_q)[: max(top_k, text_top // fetch_mult)]
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


_EXERCISE_MARKERS = (
    "brainstorming",
    "ice breakers",
    "ice breaker",
    "(a1)",
    "(a2)",
    "(a3)",
    "(a4)",
    "complete the activity",
    "discuss with your partner",
    "choose the correct",
    "multiple choice",
    "fill in the blanks",
    "match the following",
    "guess the meaning",
)


_QUERY_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "did",
        "do",
        "does",
        "how",
        "much",
        "many",
        "what",
        "when",
        "where",
        "who",
        "why",
        "which",
        "of",
        "to",
        "for",
        "in",
        "on",
        "per",
        "and",
        "or",
        "with",
        "from",
        "about",
        "this",
        "that",
        "it",
    }
)


def _query_content_terms(query: str) -> list[str]:
    toks = re.findall(r"[a-z0-9']+", (query or "").lower())
    return [t for t in toks if t not in _QUERY_STOP and len(t) > 2]


def _looks_like_exercise(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _EXERCISE_MARKERS)


def heuristic_rerank(
    text_hits: list[dict[str, Any]],
    *,
    query: str = "",
) -> list[dict[str, Any]]:
    """
    Rerank dense hits with light lexical signals.

    Dense retrieval often prefers worksheet/exercise pages that mention the same
    entity (e.g. \"astrologer\") over the narrative sentence that holds the fact.
    Boost chunks that contain distinctive query terms; demote exercise chrome.
    """
    terms = _query_content_terms(query)

    def score(h: dict[str, Any]) -> tuple:
        body = str(h.get("body") or "")
        title = str(h.get("title") or "")
        blob = f"{title}\n{body}".lower()
        is_table = 0 if str(h.get("type") or "") == "table" else 1
        exercise = 1 if _looks_like_exercise(blob) else 0
        low_info = 1 if len(body.strip()) < 300 and body.strip().endswith("?") else 0
        # Lexical overlap: rare/long terms weigh more
        lex = 0.0
        for t in terms:
            if t in blob:
                lex += 2.5 if len(t) >= 5 else 1.0
                # Extra boost when term appears near money/quantity cues
                if t in {"charge", "charged", "pies", "anna", "rupees", "cost", "price", "fee"}:
                    lex += 3.0
        # Prefer chunks that contain >=2 query terms
        hits_n = sum(1 for t in terms if t in blob)
        if hits_n >= 2:
            lex += 2.0
        if hits_n >= 3:
            lex += 2.0
        dist = float(h.get("distance") or 0.0)
        # Sort: tables first, non-exercise, non-low-info, higher lex, higher distance (IP)
        return (is_table, exercise, low_info, -lex, -dist)

    return sorted(text_hits, key=score)


def _is_pipeline_chunk(hit: dict[str, Any]) -> bool:
    """Prefer structured pipeline OKF over legacy pymupdf duplicates."""
    if str(hit.get("chunk_id") or "").strip():
        return True
    tags = hit.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_set = {str(t).lower() for t in tags}
    if "pymupdf" in tag_set:
        return False
    related = hit.get("associated_image_ids") or []
    return bool(related)


def prefer_pipeline_chunks(text_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop legacy pymupdf duplicates when a pipeline chunk covers the same
    source+page (or identical body prefix).
    """
    if not text_hits:
        return text_hits
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for h in text_hits:
        key = (str(h.get("source") or ""), int(h.get("page") or 0))
        by_page.setdefault(key, []).append(h)

    kept_ids: set[int] = set()
    out: list[dict[str, Any]] = []
    for group in by_page.values():
        pipeline = [h for h in group if _is_pipeline_chunk(h)]
        chosen = pipeline if pipeline else group
        for h in chosen:
            oid = id(h)
            if oid in kept_ids:
                continue
            kept_ids.add(oid)
            out.append(h)
    # Preserve original relative order among survivors
    order = {id(h): i for i, h in enumerate(text_hits)}
    out.sort(key=lambda h: order.get(id(h), 10**9))
    return out


def fetch_text_by_source_pages(
    source: str,
    pages: list[int],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query Milvus text collection for exact source + page set."""
    if not source or not pages:
        return []
    cfg = load_config()
    client = get_milvus_client()
    safe = source.replace('"', '\\"')
    page_list = ", ".join(str(int(p)) for p in sorted(set(pages)) if int(p) > 0)
    if not page_list:
        return []
    filt = f'source == "{safe}" and page in [{page_list}]'
    try:
        rows = client.query(
            collection_name=cfg["text_collection"],
            filter=filt,
            output_fields=TEXT_FIELDS,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_text_by_source_pages failed: %s", exc)
        return []
    return list(rows or [])


def expand_contiguous_text(
    text_hits: list[dict[str, Any]],
    *,
    window: int | None = None,
) -> list[dict[str, Any]]:
    """
    Expand around the best hit to contiguous same-source pages, sorted by
    document reading order (page, file_path). Prefers pipeline chunks.
    """
    if not text_hits:
        return text_hits
    cfg = load_config()
    win = int(window if window is not None else cfg.get("faithful_page_window", 1))
    best = text_hits[0]
    source = str(best.get("source") or "")
    center = int(best.get("page") or 0)
    if not source or center <= 0:
        return prefer_pipeline_chunks(text_hits)

    pages = list(range(max(1, center - win), center + win + 1))
    fetched = fetch_text_by_source_pages(source, pages)
    merged: dict[str, dict[str, Any]] = {}

    def _key(h: dict[str, Any]) -> str:
        return (
            str(h.get("content_hash") or "")
            or str(h.get("chunk_id") or "")
            or str(h.get("file_path") or "")
            or f"{h.get('page')}:{str(h.get('body') or '')[:80]}"
        )

    for h in text_hits:
        if str(h.get("source") or "") == source:
            merged[_key(h)] = h
    for h in fetched:
        merged[_key(h)] = h

    combined = prefer_pipeline_chunks(list(merged.values()))

    def sort_key(h: dict[str, Any]) -> tuple[int, int, str]:
        page = int(h.get("page") or 0)
        # pipeline first within page
        pipe = 0 if _is_pipeline_chunk(h) else 1
        return (page, pipe, str(h.get("file_path") or ""))

    ordered = sorted(combined, key=sort_key)
    # Keep only the chosen source pages in window
    ordered = [
        h
        for h in ordered
        if str(h.get("source") or "") == source and int(h.get("page") or 0) in set(pages)
    ]
    return ordered or prefer_pipeline_chunks(text_hits)


def fetch_images_for_pages(
    source: str,
    pages: list[int],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Load all non-excluded images for exact pages (document-faithful path)."""
    from app.chat import find_related_images

    if not source or not pages:
        return []
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in sorted(set(int(p) for p in pages if int(p) > 0)):
        # exact page: temporarily rely on proximity 0 via filter in find_related
        imgs = find_related_images(source, page, limit=limit)
        for img in imgs:
            if int(img.get("page") or 0) != page:
                continue
            ip = str(img.get("image_path") or "")
            if ip and ip not in seen:
                seen.add(ip)
                collected.append(img)
    return collected
