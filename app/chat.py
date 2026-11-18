"""RAG retrieval + rerank + dedupe + ambiguity check + Ollama chat."""

from __future__ import annotations

from typing import Any

import ollama

from app.cache_es import get_cached_answer, put_cached_answer
from app.index import embed_text, get_milvus_client
from app.logger import get_logger
from app.utils import load_config

logger = get_logger(__name__)

TEXT_OUTPUT_FIELDS = [
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
]

IMAGE_OUTPUT_FIELDS = [
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
]

SYSTEM_PROMPT_TEMPLATE = """You are an enterprise knowledge assistant. Answer using ONLY the CONTEXT below.

RULES:
1. Answer ONLY from CONTEXT; never substitute general or plausible-sounding knowledge. An invented-but-reasonable answer is a worse failure than refusing.
2. Preserve markdown tables EXACTLY; if a table lists items, use exactly those items, never a generic substitute list.
3. Every context passage is pre-tagged with its own citation. When you use a passage, copy that passage's own citation tag verbatim. Never literally print the words "source", "source_loc", or "section" as field-name text.
4. RELATED IMAGES are page-mapped evidence. Only reference an image when it belongs to the same source/page as the text you used. Include its image_path as a markdown image. Never invent images. Never use header/footer/watermark images.
5. If context only contains the question itself, or doesn't contain the specific answer, respond exactly "Not found in approved OKF bundle" — never guess.
6. When a single passage directly answers the question (e.g. an FAQ entry), quote it word-for-word in full — no paraphrasing, shortening, or dropping sentences. Summarizing across passages is only for genuine multi-passage synthesis.
7. Conversation history is for pronoun/reference resolution ONLY, never a source of facts — every factual claim must still come from CONTEXT.
8. Never narrate retrieval process, confidence, or uncertainty ("the context does not explicitly state...", "it is mentioned that..."); either answer directly per rule 6, or give the exact refusal line in rule 5 — nothing else, no meta-commentary.
{ambiguity_rule}

CONTEXT:
{context}
{images_context}
"""

AMBIGUITY_RULE = (
    "9. The retrieved context spans multiple distinct unrelated documents and the "
    "question does not clearly point to one. Ask ONE specific clarifying question "
    "naming these candidate sources/titles, instead of guessing or blending them:\n"
    "{candidates}"
)


def format_citation(payload: dict[str, Any]) -> str:
    """Format a citation tag from a Milvus hit payload."""
    source = str(payload.get("source") or "").strip()
    source_loc = str(payload.get("source_loc") or "").strip()
    version = str(payload.get("version") or "").strip()
    section = str(payload.get("section") or "").strip()

    if not version and not section:
        return f"{source} {source_loc}".strip()

    parts = [source]
    if version:
        parts.append(version)
    if source_loc:
        parts.append(source_loc)
    if section:
        parts.append(f"Section {section}")
    return " | ".join(p for p in parts if p)


def _is_low_information(hit: dict[str, Any]) -> bool:
    body = str(hit.get("body") or "").strip()
    tags = hit.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_set = {str(t).lower() for t in tags}
    if "section_header" in tag_set or "title" in tag_set:
        return True
    if len(body) < 300 and body.endswith("?"):
        return True
    return False


def _rerank(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tables first, then demote low-information chunks within their bucket."""

    def key(h: dict[str, Any]) -> tuple[int, int]:
        is_table = 0 if str(h.get("type") or "") == "table" else 1
        low_info = 1 if _is_low_information(h) else 0
        return (is_table, low_info)

    return sorted(hits, key=key)


def _dedupe_by_file(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse exact duplicate rows by file_path; keep first occurrence."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in hits:
        fp = str(h.get("file_path") or "")
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        out.append(h)
    return out


def _detect_ambiguous_topics(
    hits: list[dict[str, Any]],
    threshold: float,
    probe_top_n: int = 3,
) -> list[dict[str, str]]:
    """Return candidate {source, title} list when top hits compete across sources."""
    if len(hits) < 2:
        return []

    probe = hits[:probe_top_n]
    best = probe[0]
    best_source = str(best.get("source") or "")
    if all(str(h.get("source") or "") == best_source for h in probe):
        return []

    rival = None
    for h in probe[1:]:
        if str(h.get("source") or "") != best_source:
            rival = h
            break
    if rival is None:
        return []

    best_dist = float(best.get("distance", 0.0))
    rival_dist = float(rival.get("distance", 0.0))
    gap = best_dist - rival_dist
    if gap > threshold:
        return []

    candidates: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for h in probe:
        src = str(h.get("source") or "")
        if src in seen_sources:
            continue
        seen_sources.add(src)
        candidates.append(
            {
                "source": src,
                "title": str(h.get("title") or ""),
            }
        )
    return candidates


def search_milvus(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Embed query (with BGE instruction prefix) and search the text collection."""
    cfg = load_config()
    client = get_milvus_client()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = embed_text(prefixed)
    results = client.search(
        collection_name=cfg["text_collection"],
        data=[vector],
        limit=top_k,
        output_fields=TEXT_OUTPUT_FIELDS,
    )
    hits: list[dict[str, Any]] = []
    # pymilvus MilvusClient returns list[list[dict]]
    rows = results[0] if results else []
    for row in rows:
        entity = row.get("entity") or row
        payload = {k: entity.get(k) for k in TEXT_OUTPUT_FIELDS if k in entity}
        # Also copy any present fields from row root
        for k in TEXT_OUTPUT_FIELDS:
            if k not in payload or payload[k] is None:
                if k in row:
                    payload[k] = row[k]
        payload["distance"] = float(row.get("distance", row.get("score", 0.0)))
        payload["id"] = row.get("id")
        hits.append(payload)
    return hits


def _is_header_footer_image(img: dict[str, Any]) -> bool:
    if img.get("excluded") is True:
        return True
    if str(img.get("image_role") or "").lower() in {"header_footer", "header", "footer"}:
        return True
    tags = img.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_set = {str(t).lower() for t in tags}
    return bool(tag_set & {"header_footer", "header", "footer", "watermark"})


def find_related_images(
    source: str,
    page: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Query image collection by source + page proximity; drop excluded/header/footer."""
    cfg = load_config()
    proximity = int(cfg.get("page_proximity_range", 1))
    client = get_milvus_client()
    lo = max(0, int(page) - proximity)
    hi = int(page) + proximity
    safe_source = source.replace('"', '\\"')
    filt = (
        f'source == "{safe_source}" and page >= {lo} and page <= {hi} '
        f"and (excluded == false or excluded is null)"
    )
    try:
        rows = client.query(
            collection_name=cfg["image_collection"],
            filter=filt,
            output_fields=IMAGE_OUTPUT_FIELDS,
            limit=max(limit * 3, 20),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("find_related_images query failed: %s", exc)
        return []

    filtered = [r for r in rows if not _is_header_footer_image(r)]
    for r in filtered:
        r["_page_diff"] = abs(int(r.get("page") or 0) - int(page))
        r["mapped_from_page"] = int(page)
    filtered.sort(key=lambda r: (r.get("_page_diff", 0), str(r.get("image_path") or "")))
    return filtered[:limit]


def _collect_related_images(
    hits: list[dict[str, Any]],
    max_images: int,
    probe_top_n: int = 3,
) -> list[dict[str, Any]]:
    if not hits or max_images <= 0:
        return []

    best_source = str(hits[0].get("source") or "")
    if not best_source:
        return []

    same_source = [
        h for h in hits[:probe_top_n] if str(h.get("source") or "") == best_source
    ]
    page_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for h in same_source:
        key = (best_source, int(h.get("page") or 0))
        if key not in seen:
            seen.add(key)
            page_keys.append(key)

    buckets: list[list[dict[str, Any]]] = []
    for src, page in page_keys:
        buckets.append(find_related_images(src, page, limit=10))

    # Round-robin across buckets
    collected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    cursors = [0] * len(buckets)
    while len(collected) < max_images and buckets:
        progressed = False
        for i, bucket in enumerate(buckets):
            while cursors[i] < len(bucket):
                img = bucket[cursors[i]]
                cursors[i] += 1
                ip = str(img.get("image_path") or "")
                if ip and ip not in seen_paths:
                    seen_paths.add(ip)
                    collected.append(img)
                    progressed = True
                    break
            if len(collected) >= max_images:
                break
        if not progressed:
            break
    return collected


def _build_context(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for h in hits:
        citation = format_citation(h)
        ctype = h.get("type") or ""
        title = h.get("title") or ""
        body = str(h.get("body") or "")[:4000]
        parts.append(
            f"\n---\n[{citation}] Type:{ctype} Title:{title}\n{body}\n"
        )
    return "".join(parts)


def _build_images_context(images: list[dict[str, Any]]) -> str:
    if not images:
        return ""
    lines = [
        "\nRELATED IMAGES (page-mapped evidence — only use with matching source/page text):"
    ]
    for img in images:
        lines.append(
            f"- image_path: {img.get('image_path')} | source: {img.get('source')} | "
            f"page: {img.get('page')} | mapped_from_text_page: {img.get('mapped_from_page')} | "
            f"caption: {img.get('caption')}"
        )
    return "\n".join(lines) + "\n"


def _build_evidence_map(
    hits: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map each used text hit to same-page images for the API/UI."""
    evidence: list[dict[str, Any]] = []
    for h in hits[:3]:
        page = int(h.get("page") or 0)
        source = str(h.get("source") or "")
        page_images = [
            {
                "image_path": img.get("image_path"),
                "caption": img.get("caption"),
                "page": img.get("page"),
            }
            for img in images
            if str(img.get("source") or "") == source
            and abs(int(img.get("page") or 0) - page) <= int(
                load_config().get("page_proximity_range", 1)
            )
        ]
        evidence.append(
            {
                "citation": format_citation(h),
                "source": source,
                "page": page,
                "type": h.get("type"),
                "title": h.get("title"),
                "images": page_images,
            }
        )
    return evidence


def _build_search_query(query: str, history: list[dict[str, str]] | None) -> str:
    if not history:
        return query
    prior_user = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            prior_user = msg.get("content") or ""
            break
    if prior_user:
        return f"{prior_user} {query}"
    return query


def _build_verbatim_excerpt(answer: str, hits: list[dict[str, Any]]) -> str:
    if "not found in approved okf bundle" in answer.lower():
        return answer
    if not hits:
        return answer
    best = hits[0]
    citation = format_citation(best)
    body = str(best.get("body") or "")
    return (
        f"{answer}\n\n---\n**Exact source text** [{citation}]:\n\n{body}"
    )


def _chat_with_ollama(messages: list[dict[str, str]]) -> str:
    cfg = load_config()
    client = ollama.Client(host=cfg["ollama_url"])
    options = {
        "temperature": 0.1,
        "num_ctx": int(cfg.get("chat_num_ctx", 8192)),
        "num_predict": int(cfg.get("chat_num_predict", 2048)),
    }
    models = [cfg.get("chat_model"), cfg.get("chat_model_fallback")]
    last_exc: Exception | None = None
    for model in models:
        if not model:
            continue
        try:
            resp = client.chat(model=model, messages=messages, options=options)
            if isinstance(resp, dict):
                content = (
                    (resp.get("message") or {}).get("content")
                    or resp.get("response")
                    or ""
                )
            else:
                msg = getattr(resp, "message", None)
                content = (
                    (getattr(msg, "content", None) if msg is not None else None)
                    or getattr(resp, "response", None)
                    or ""
                )
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Ollama chat with %s failed: %s", model, exc)
    logger.error("All chat models unavailable: %s", last_exc)
    return (
        "The chat model is currently unavailable. Please verify Ollama is "
        "running and the configured chat model is pulled, then try again."
    )


def rag_answer(
    query: str,
    top_k: int = 10,
    history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    user_id: str = "anonymous",
    access_scopes: list[str] | None = None,
) -> dict[str, Any]:
    """Full RAG pipeline via orchestrator (hybrid retrieve + grounded answer)."""
    from app.orchestrator import rag_answer as _orchestrated

    return _orchestrated(
        query,
        top_k=top_k,
        history=history,
        conversation_id=conversation_id,
        user_id=user_id,
        access_scopes=access_scopes,
    )
