"""RAG orchestration: retrieve → assemble → optional VLM → grounded answer."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import ollama

from app.cache_es import get_cached_answer, put_cached_answer
from app.db.models import Conversation, Message
from app.db.session import get_session, init_db
from app.chat import AMBIGUITY_RULE, _detect_ambiguous_topics
from app.faithful import (
    build_faithful_answer,
    load_okf_images_for_pages,
    looks_like_procedure,
    query_wants_procedure,
    select_focused_hits,
)
from app.logger import get_logger
from app.models_registry import get_model_registry
from app.query_modes import (
    detect_query_mode,
    estimate_confidence,
    mode_prompt_addendum,
    suggested_questions,
)
from app.retrieval import (
    expand_contiguous_text,
    fetch_images_for_pages,
    heuristic_rerank,
    hybrid_retrieve,
    prefer_pipeline_chunks,
)
from app.utils import load_config, okf_bundle_dir

logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an enterprise knowledge assistant. Answer using ONLY the CONTEXT below.

RULES:
1. Answer ONLY from CONTEXT; never substitute general or plausible-sounding knowledge.
2. Preserve markdown tables EXACTLY.
3. Every context passage is pre-tagged with its citation. Copy citation tags verbatim when used.
4. RELATED IMAGES are evidence. Only reference an image when it belongs to the same source/page as used text. Include image_path as a markdown image. Never invent images.
5. Refuse with exactly "Not found in approved OKF bundle" ONLY when CONTEXT has no passage that supports the question. If CONTEXT describes the facts (even without using the question's wording), you MUST answer.
6. For list / discuss / qualities / "remarkable" / "exceptional" questions: synthesize a clear structured answer (short intro + numbered points + brief conclusion) that is fully entailed by CONTEXT. Do not refuse merely because the document does not already print a numbered list.
7. Prefer a ChatGPT-style grounded answer: organize facts from CONTEXT; quote sparingly; do not dump the entire passage unless the user asks for the exact text.
8. Conversation history is for reference resolution ONLY — facts must come from CONTEXT.
9. Distinguish explicit document facts from visual interpretations marked as [VLM].
10. If sources conflict, state the conflict with citations.
11. Bullet formatting: when the source uses checkbox/tick markers (, ✓, ✔, ☑, □, ■, or similar box/tick glyphs) at the start of list items, render each item as a solid round bullet using "• " (U+2022). Do not keep box or checkmark glyphs in the answer. Keep the item text unchanged.
12. End with a short Reference line using the citation tags (source + page) from CONTEXT.
{mode_addendum}
{ambiguity_rule}

CONTEXT:
{context}
{images_context}
{vlm_context}
"""


_LIT_EXPAND_RE = re.compile(
    r"(?i)\b(story|passage|qualities|exceptional|remarkable|discuss|list the|"
    r"in the (?:story|passage|text)|astrologer|town hall)\b"
)


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
            f"page: {img.get('page')} | figure_label: {img.get('figure_label') or ''} | "
            f"caption: {img.get('caption') or img.get('image_description')}"
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


_THINK_RE = re.compile(
    r"(?is)<think>.*?</think>|<thinking>.*?</thinking>|^\s*Thinking:.*?(?=\n\n|\Z)"
)
_INCOMPLETE_END_RE = re.compile(
    r"(?i)\b(a|an|the|by|to|of|for|with|and|or|into|from|that|which|who|whom|"
    r"whose|when|where|while|because|as|but|than|then|this|these|those|its|"
    r"his|her|their|our|my)\s*$"
)


def _strip_model_chrome(text: str) -> str:
    """Remove reasoning wrappers some Ollama models emit before the real answer."""
    cleaned = _THINK_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _looks_incomplete(answer: str) -> bool:
    """True when generation likely hit num_predict mid-sentence."""
    text = (answer or "").strip()
    if len(text) < 40:
        return False
    if text.endswith(("...", "…")):
        return True
    # Ends without sentence punctuation and looks cut off
    if re.search(r"[.!?…\"”)]\s*$", text):
        return False
    if _INCOMPLETE_END_RE.search(text):
        return True
    # Trailing lowercase letter / open quote / dangling comma — usually truncated
    if re.search(r"[,:;]\s*$", text):
        return True
    if re.search(r"[a-z0-9]\s*$", text) and not re.search(
        r"(?i)\b(bundle|reference|sources?|citations?)\s*$", text
    ):
        return True
    return False


def _extract_chat_content(resp: Any) -> tuple[str, str]:
    """Return (content, done_reason) from an Ollama chat response."""
    done_reason = ""
    if isinstance(resp, dict):
        content = str(
            (resp.get("message") or {}).get("content") or resp.get("response") or ""
        )
        done_reason = str(resp.get("done_reason") or resp.get("doneReason") or "")
        return content.strip(), done_reason
    msg = getattr(resp, "message", None)
    content = str(
        (getattr(msg, "content", None) if msg else None)
        or getattr(resp, "response", None)
        or ""
    )
    done_reason = str(getattr(resp, "done_reason", "") or "")
    return content.strip(), done_reason


def _chat(messages: list[dict[str, str]]) -> str:
    cfg = load_config()
    client = ollama.Client(host=cfg["ollama_url"])
    num_predict = int(cfg.get("chat_num_predict", 2048))
    options = {
        "temperature": 0.1,
        "num_ctx": int(cfg.get("chat_num_ctx", 8192)),
        "num_predict": num_predict,
    }
    for model in [cfg.get("chat_model"), cfg.get("chat_model_fallback")]:
        if not model:
            continue
        try:
            resp = client.chat(model=model, messages=messages, options=options)
            content, done_reason = _extract_chat_content(resp)
            answer = _strip_model_chrome(content)
            # Continue once if Ollama stopped for length / mid-sentence cut-off
            if answer and (
                done_reason == "length" or _looks_incomplete(answer)
            ):
                cont_msgs = list(messages) + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was cut off mid-sentence. "
                            "Continue exactly from where you stopped. Do not "
                            "repeat earlier sentences; finish the full answer "
                            "with conclusion and Reference."
                        ),
                    },
                ]
                cont_opts = dict(options)
                cont_opts["num_predict"] = max(num_predict, 1024)
                try:
                    cont_resp = client.chat(
                        model=model, messages=cont_msgs, options=cont_opts
                    )
                    cont_text, _ = _extract_chat_content(cont_resp)
                    cont_text = _strip_model_chrome(cont_text)
                    if cont_text:
                        # If model restarted instead of continuing, prefer longer complete text
                        if cont_text.startswith(answer[:80]):
                            answer = cont_text
                        else:
                            answer = (answer.rstrip() + " " + cont_text.lstrip()).strip()
                except Exception as cont_exc:  # noqa: BLE001
                    logger.warning("Answer continuation failed for %s: %s", model, cont_exc)
            return answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat model %s failed: %s", model, exc)
    return (
        "The chat model is currently unavailable. Please verify Ollama is running "
        "and the configured chat model is pulled, then try again."
    )


def re_find_pdfs(text: str) -> list[str]:
    return re.findall(r"[\w.\-]+\.pdf", text, flags=re.I)


def _validate_citations(
    answer: str, hits: list[dict[str, Any]], images: list[dict[str, Any]]
) -> list[str]:
    warnings: list[str] = []
    known_sources = {str(h.get("source") or "") for h in hits}
    known_sources |= {str(i.get("source") or "") for i in images}
    for m in re_find_pdfs(answer):
        if m not in known_sources and known_sources:
            warnings.append(f"unverified_source:{m}")
    return warnings


def _build_evidence_map(
    text_hits: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cfg = load_config()
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
                        "figure_label": img.get("figure_label"),
                        "page": img.get("page"),
                    }
                    for img in images
                    if str(img.get("source") or "") == source
                    and abs(int(img.get("page") or 0) - page)
                    <= int(cfg.get("page_proximity_range", 1))
                ],
            }
        )
    return evidence_map


def rag_answer(
    query: str,
    *,
    top_k: int = 10,
    history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    user_id: str = "anonymous",
    access_scopes: list[str] | None = None,
    mode: str | None = None,
    skip_cache: bool = False,
) -> dict[str, Any]:
    init_db()
    if not skip_cache:
        cached = get_cached_answer(query, history)
        if cached is not None:
            return cached

    cfg = load_config()
    max_chars = int(cfg.get("max_context_chars", 24000))
    max_images = int(cfg.get("max_images_per_answer", 4))
    history_turns = int(cfg.get("chat_history_turns", 6))
    faithful_max = int(cfg.get("faithful_max_images", 12))
    mode_detect = bool(cfg.get("query_mode_detection", True))
    force_mode = mode if mode in {"qa", "mcq", "summary", "procedure"} else None
    query_mode = detect_query_mode(query, force=force_mode if mode_detect else "qa")  # type: ignore[arg-type]

    retrieved = hybrid_retrieve(
        query, top_k=top_k, history=history, access_scopes=access_scopes
    )
    text_hits = prefer_pipeline_chunks(
        heuristic_rerank(
            retrieved["text_hits"],
            query=str(retrieved.get("search_query") or query or ""),
        )
    )

    best_source = str(text_hits[0].get("source") or "") if text_hits else ""
    allowed_sources = {best_source} if best_source else set()

    # Focused faithful assembly: best page only (optional small window)
    use_faithful = (
        bool(cfg.get("document_faithful_answers", True))
        and bool(text_hits)
        and query_mode in {"procedure", "qa"}
    )
    if use_faithful:
        win = int(cfg.get("faithful_page_window", 0))
        max_pages = int(cfg.get("faithful_max_pages", 1))
        # Literature / list-discuss questions often span a page break (name on p16,
        # qualities on p17) — widen the focus window so CONTEXT is complete.
        if query_mode == "qa" and _LIT_EXPAND_RE.search(query or ""):
            win = max(win, 1)
            max_pages = max(max_pages, 2)
        # Score-focus first so neighboring Advanced Search pages are not pulled in
        text_hits = select_focused_hits(text_hits, query, max_pages=max(1, max_pages))
        if win > 0:
            text_hits = expand_contiguous_text(text_hits, window=win)
            text_hits = select_focused_hits(text_hits, query, max_pages=max(1, max_pages))
        pages = sorted({int(h.get("page") or 0) for h in text_hits if int(h.get("page") or 0) > 0})
        images = fetch_images_for_pages(best_source, pages, limit=faithful_max)
        if not images:
            images = load_okf_images_for_pages(best_source, pages)
        else:
            seen = {str(i.get("image_path") or "") for i in images}
            for extra in load_okf_images_for_pages(best_source, pages):
                ip = str(extra.get("image_path") or "")
                if ip and ip not in seen:
                    images.append(extra)
                    seen.add(ip)
    else:
        images = [
            img
            for img in (retrieved.get("expanded_images") or [])
            if str(img.get("source") or "") in allowed_sources
        ]
        for item in retrieved.get("fused") or []:
            if item["modality"] != "image":
                continue
            p = item["payload"]
            if str(p.get("source") or "") not in allowed_sources:
                continue
            ip = str(p.get("image_path") or "")
            if ip and all(str(i.get("image_path")) != ip for i in images):
                images.append(p)
            if len(images) >= max_images:
                break
        images = images[:max_images]

    images = [i for i in images if str(i.get("source") or "") in allowed_sources or not allowed_sources]

    # Prefer reconstruction for procedures
    faithful = None
    if use_faithful and query_mode == "procedure":
        procedural_evidence = any(
            looks_like_procedure(str(h.get("body") or "")) for h in text_hits
        ) or query_wants_procedure(query)
        if procedural_evidence:
            faithful = build_faithful_answer(text_hits, images, query=query)
    elif use_faithful and query_mode == "qa":
        # Keep prior behavior: faithful when procedural evidence exists
        procedural_evidence = any(
            looks_like_procedure(str(h.get("body") or "")) for h in text_hits
        ) or query_wants_procedure(query)
        if procedural_evidence:
            faithful = build_faithful_answer(text_hits, images, query=query)

    answer_mode = "llm"
    steps: list[dict[str, Any]] = []
    images_interleaved = False

    if faithful and faithful.get("answer"):
        answer = str(faithful["answer"])
        images = list(faithful.get("images") or images)
        steps = list(faithful.get("steps") or [])
        answer_mode = "document_faithful"
        images_interleaved = True
        query_mode = "procedure"
    else:
        context = _build_context(text_hits, max_chars)
        images_context = _build_images_context(images[:max_images])
        # Skip VLM on faithful path; for LLM path respect config (default off recommended)
        vlm_context = _vlm_interpret(
            images, query, int(cfg.get("answer_vlm_max_images", 2))
        )

        ambiguity_rule = ""
        threshold = float(cfg.get("ambiguity_score_gap_threshold", 0.03))
        if query_mode == "qa" and text_hits:
            candidates = _detect_ambiguous_topics(text_hits, threshold)
            if candidates:
                lines = "\n".join(
                    f"- {c.get('source') or ''}: {c.get('title') or ''}" for c in candidates
                )
                ambiguity_rule = AMBIGUITY_RULE.format(candidates=lines)

        addendum = mode_prompt_addendum(query_mode)  # type: ignore[arg-type]
        mode_block = f"\n{addendum}\n" if addendum else ""

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            mode_addendum=mode_block,
            ambiguity_rule=ambiguity_rule,
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
        answer_mode = query_mode
        # Local models sometimes refuse even when CONTEXT clearly supports a
        # list/discuss question — one grounded retry with an explicit synthesis ask.
        if (
            text_hits
            and query_mode == "qa"
            and "not found in approved okf bundle" in (answer or "").lower()
            and _LIT_EXPAND_RE.search(query or "")
        ):
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "CONTEXT does support this question. Using only CONTEXT, write a "
                        "structured answer with a short intro, numbered exceptional "
                        "qualities/facts, a one-line conclusion, and a Reference citation. "
                        "Do not answer Not found."
                    ),
                }
            ]
            retry = _chat(retry_messages)
            if retry and "not found in approved okf bundle" not in retry.lower():
                answer = retry
        append_exact = bool(cfg.get("append_exact_source_excerpt", False))
        if (
            append_exact
            and text_hits
            and "not found in approved okf bundle" not in answer.lower()
            and query_mode != "summary"
            and "clarifying" not in answer.lower()
            and not answer.strip().endswith("?")
        ):
            best = text_hits[0]
            answer = (
                f"{answer}\n\n---\n**Exact source text** [{format_citation(best)}]:\n\n"
                f"{str(best.get('body') or '')}"
            )

    try:
        from app.rich_text import normalize_list_markers

        answer = normalize_list_markers(answer)
    except Exception:  # noqa: BLE001
        pass

    warnings = _validate_citations(answer, text_hits, images)
    evidence_map = _build_evidence_map(text_hits, images)
    confidence = estimate_confidence(
        answer=answer, text_hits=text_hits, warnings=warnings, mode=answer_mode
    )
    suggestions = suggested_questions(
        query, text_hits, mode=query_mode, limit=4
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
        "steps": steps,
        "mode": answer_mode,
        "query_mode": query_mode,
        "confidence": confidence,
        "suggested_questions": suggestions,
        "images_interleaved": images_interleaved,
        "intent": retrieved.get("intent"),
        "warnings": warnings,
        "cache": "miss",
        "conversation_id": conversation_id,
    }

    try:
        with get_session() as session:
            conv = None
            if conversation_id:
                conv = session.get(Conversation, conversation_id)
            if conv is None:
                conv = Conversation(
                    id=conversation_id or str(uuid4()), user_id=user_id, title=query[:80]
                )
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
                        "mode": answer_mode,
                        "query_mode": query_mode,
                        "confidence": confidence,
                        "suggested_questions": suggestions,
                        "steps": steps,
                    },
                )
            )
            result["conversation_id"] = conv.id
            # Expose last assistant message id for feedback
            result["message_id"] = None
            try:
                last = (
                    session.query(Message)
                    .filter_by(conversation_id=conv.id, role="assistant")
                    .order_by(Message.created_at.desc())
                    .first()
                )
                if last:
                    result["message_id"] = last.id
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Conversation persist failed: %s", exc)

    if not skip_cache:
        put_cached_answer(query, result, history)
    return result
