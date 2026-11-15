"""RAG orchestration: retrieve → assemble → optional VLM → grounded answer."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ollama

from app.cache_es import get_cached_answer, put_cached_answer
from app.db.models import Conversation, Message
from app.db.session import get_session, init_db
from app.logger import get_logger
from app.models_registry import get_model_registry
from app.retrieval import heuristic_rerank, hybrid_retrieve
from app.utils import load_config, okf_bundle_dir

logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an enterprise knowledge assistant. Answer using ONLY the CONTEXT below.

RULES:
1. Answer ONLY from CONTEXT; never substitute general or plausible-sounding knowledge.
2. Preserve markdown tables EXACTLY.
3. Every context passage is pre-tagged with its citation. Copy citation tags verbatim when used.
4. RELATED IMAGES are evidence. Only reference an image when it belongs to the same source/page as used text. Include image_path as a markdown image. Never invent images.
5. If evidence is insufficient, respond exactly: "Not found in approved OKF bundle"
6. When a single passage directly answers, quote it fully.
7. Conversation history is for reference resolution ONLY — facts must come from CONTEXT.
8. Distinguish explicit document facts from visual interpretations marked as [VLM].
9. If sources conflict, state the conflict with citations.
{ambiguity_rule}

CONTEXT:
{context}
{images_context}
{vlm_context}
"""


def format_citation(payload: dict[str, Any]) -> str:
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


def _build_context(hits: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for h in hits:
        citation = format_citation(h)
        block = (
            f"\n---\n[{citation}] Type:{h.get('type') or ''} Title:{h.get('title') or ''}\n"
            f"{str(h.get('body') or '')[:4000]}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)


def _build_images_context(images: list[dict[str, Any]]) -> str:
    if not images:
        return ""
    lines = ["\nRELATED IMAGES:"]
    for img in images:
        lines.append(
            f"- image_path: {img.get('image_path')} | source: {img.get('source')} | "
            f"page: {img.get('page')} | caption: {img.get('caption') or img.get('image_description')}"
        )
    return "\n".join(lines) + "\n"


def _vlm_interpret(images: list[dict[str, Any]], query: str, max_images: int) -> str:
    cfg = load_config()
    if not cfg.get("answer_vlm_enabled", True) or max_images <= 0:
        return ""
    registry = get_model_registry()
    vision = registry.specs()["vision"].name
    if not vision:
        return ""
    client = ollama.Client(host=cfg["ollama_url"])
    blocks: list[str] = []
    for img in images[:max_images]:
        rel = str(img.get("image_path") or "")
        if not rel:
            continue
        abs_path = okf_bundle_dir() / rel
        if not abs_path.exists():
            continue
        prompt = (
            f"User question: {query}\n"
            "Inspect this document figure. Describe only what is visible that helps answer. "
            "Do not invent labels or numbers you cannot read."
        )
        try:
            resp = client.generate(model=vision, prompt=prompt, images=[str(abs_path)])
            text = (
                (resp.get("response") if isinstance(resp, dict) else getattr(resp, "response", ""))
                or ""
            ).strip()
            if text:
                blocks.append(
                    f"[VLM] source={img.get('source')} page={img.get('page')} path={rel}\n{text}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM interpret failed for %s: %s", rel, exc)
    if not blocks:
        return ""
    return "\nVLM INTERPRETATIONS:\n" + "\n\n".join(blocks) + "\n"


def _chat(messages: list[dict[str, str]]) -> str:
    cfg = load_config()
    client = ollama.Client(host=cfg["ollama_url"])
    options = {
        "temperature": 0.1,
        "num_ctx": int(cfg.get("chat_num_ctx", 4096)),
        "num_predict": int(cfg.get("chat_num_predict", 512)),
    }
    for model in [cfg.get("chat_model"), cfg.get("chat_model_fallback")]:
        if not model:
            continue
        try:
            resp = client.chat(model=model, messages=messages, options=options)
            if isinstance(resp, dict):
                return str(
                    (resp.get("message") or {}).get("content") or resp.get("response") or ""
                ).strip()
            msg = getattr(resp, "message", None)
            return str(
                (getattr(msg, "content", None) if msg else None)
                or getattr(resp, "response", None)
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat model %s failed: %s", model, exc)
    return (
        "The chat model is currently unavailable. Please verify Ollama is running "
        "and the configured chat model is pulled, then try again."
    )


def _validate_citations(answer: str, hits: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[str]:
    """Return warnings if answer invents sources/pages not in evidence."""
    warnings: list[str] = []
    known_sources = {str(h.get("source") or "") for h in hits}
    known_sources |= {str(i.get("source") or "") for i in images}
    # Soft check: if answer claims a .pdf not in evidence
    for m in re_find_pdfs(answer):
        if m not in known_sources and known_sources:
            warnings.append(f"unverified_source:{m}")
    return warnings


def re_find_pdfs(text: str) -> list[str]:
    import re

    return re.findall(r"[\w.\-]+\.pdf", text, flags=re.I)


def rag_answer(
    query: str,
    *,
    top_k: int = 10,
    history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    user_id: str = "anonymous",
    access_scopes: list[str] | None = None,
) -> dict[str, Any]:
    init_db()
    cached = get_cached_answer(query, history)
    if cached is not None:
        return cached

    cfg = load_config()
    max_chars = int(cfg.get("max_context_chars", 24000))
    max_images = int(cfg.get("max_images_per_answer", 4))
    history_turns = int(cfg.get("chat_history_turns", 6))

    retrieved = hybrid_retrieve(
        query, top_k=top_k, history=history, access_scopes=access_scopes
    )
    text_hits = heuristic_rerank(retrieved["text_hits"])
    images = retrieved.get("expanded_images") or []
    # Prefer fused image hits that are relevant
    for item in retrieved.get("fused") or []:
        if item["modality"] == "image":
            p = item["payload"]
            ip = str(p.get("image_path") or "")
            if ip and all(str(i.get("image_path")) != ip for i in images):
                images.append(p)
            if len(images) >= max_images:
                break
    images = images[:max_images]

    context = _build_context(text_hits, max_chars)
    images_context = _build_images_context(images)
    vlm_context = _vlm_interpret(
        images, query, int(cfg.get("answer_vlm_max_images", 2))
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        ambiguity_rule="",
        context=context or "(no matching passages)",
        images_context=images_context,
        vlm_context=vlm_context,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        trimmed = [m for m in history if m.get("role") in {"user", "assistant"}][
            -history_turns:
        ]
        messages.extend(trimmed)
    messages.append({"role": "user", "content": query})

    answer = _chat(messages)
    if text_hits and "not found in approved okf bundle" not in answer.lower():
        best = text_hits[0]
        answer = (
            f"{answer}\n\n---\n**Exact source text** [{format_citation(best)}]:\n\n"
            f"{str(best.get('body') or '')}"
        )

    warnings = _validate_citations(answer, text_hits, images)
    evidence_map = []
    for h in text_hits[:3]:
        page = int(h.get("page") or 0)
        source = str(h.get("source") or "")
        evidence_map.append(
            {
                "citation": format_citation(h),
                "source": source,
                "page": page,
                "type": h.get("type"),
                "title": h.get("title"),
                "images": [
                    {
                        "image_path": img.get("image_path"),
                        "caption": img.get("caption") or img.get("image_description"),
                        "page": img.get("page"),
                    }
                    for img in images
                    if str(img.get("source") or "") == source
                    and abs(int(img.get("page") or 0) - page)
                    <= int(cfg.get("page_proximity_range", 1))
                ],
            }
        )

    result = {
        "answer": answer,
        "sources": [
            {k: text_hits[0].get(k) for k in text_hits[0] if k != "vector"}
        ]
        if text_hits
        else [],
        "images": images,
        "evidence_map": evidence_map,
        "intent": retrieved.get("intent"),
        "warnings": warnings,
        "cache": "miss",
        "conversation_id": conversation_id,
    }

    # Persist conversation if requested
    try:
        with get_session() as session:
            conv = None
            if conversation_id:
                conv = session.get(Conversation, conversation_id)
            if conv is None:
                conv = Conversation(id=conversation_id or str(uuid4()), user_id=user_id, title=query[:80])
                session.add(conv)
                session.flush()
            session.add(Message(conversation_id=conv.id, role="user", content=query))
            session.add(
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
                    evidence={
                        "sources": result["sources"],
                        "images": [
                            {"image_path": i.get("image_path"), "page": i.get("page")}
                            for i in images
                        ],
                    },
                )
            )
            result["conversation_id"] = conv.id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Conversation persist failed: %s", exc)

    put_cached_answer(query, result, history)
    return result
