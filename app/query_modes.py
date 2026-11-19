"""Query mode detection and mode-specific prompt addenda for Phase 1 MVP."""

from __future__ import annotations

import re
from typing import Any, Literal

from app.faithful import query_wants_procedure

QueryMode = Literal["qa", "mcq", "summary", "procedure"]

_MCQ_OPTION_RE = re.compile(
    r"(?im)^\s*(?:\(?([A-Da-d]|[1-4])\)?[.)]\s+|\b([A-Da-d])\s*[-–:]\s+).+"
)
_MCQ_INLINE_RE = re.compile(
    r"(?i)\b(?:options?|choose|select|which of the following|mcq|true\s*/\s*false|"
    r"true or false|correct (?:answer|option|statement))\b"
)
_SUMMARY_RE = re.compile(
    r"(?i)\b(?:summar(?:y|ise|ize)|summarisation|summarization|key points?|"
    r"important points?|executive summary|overview|in brief|bullet[- ]?point)\b"
)


MCQ_PROMPT_ADDENDUM = """
MODE: MCQ / True-False
- Identify the correct option(s) using ONLY CONTEXT evidence.
- For single-correct questions: state the correct option clearly (e.g. "Correct answer: B").
- For multi-correct: list every option supported by CONTEXT.
- For True/False: state True or False with a one-sentence justification from CONTEXT.
- Briefly explain why other options are incorrect when CONTEXT supports that judgment.
- If CONTEXT cannot verify any option, respond exactly: "Not found in approved OKF bundle"
- Cite sources with the pre-tagged citation labels.
""".strip()

SUMMARY_PROMPT_ADDENDUM = """
MODE: SUMMARIZATION
- Produce a grounded summary using ONLY CONTEXT (no outside knowledge).
- Prefer concise bullet points when the user asks for bullets/key points; otherwise a short paragraph summary.
- Cover only facts present in CONTEXT; omit speculation.
- If CONTEXT is insufficient, respond exactly: "Not found in approved OKF bundle"
- Include citation tags for major claims.
""".strip()

QA_PROMPT_ADDENDUM = """
MODE: QUESTION ANSWERING
- Give a direct, detailed answer grounded only in CONTEXT.
- Prefer multi-passage synthesis when several sections are needed.
- For textbook-style prompts (list / discuss / qualities / remarkable place): produce a ChatGPT-like structured answer from CONTEXT facts — numbered points with brief evidence — even if CONTEXT never uses those headings.
- Do not dump the full passage; organize and paraphrase tightly while staying faithful.
""".strip()

PROCEDURE_NOTE = "procedure"  # handled by faithful path; prompt unused


def detect_query_mode(query: str, *, force: QueryMode | None = None) -> QueryMode:
    if force in {"qa", "mcq", "summary", "procedure"}:
        return force
    q = (query or "").strip()
    if not q:
        return "qa"
    # Explicit summary / MCQ cues win over broad procedure keywords (e.g. "insert")
    if _SUMMARY_RE.search(q):
        return "summary"
    if _MCQ_INLINE_RE.search(q) or len(_MCQ_OPTION_RE.findall(q)) >= 2:
        return "mcq"
    # Heuristic: looks like a stem + lettered options pasted in
    if re.search(r"(?im)^\s*[A-D][.)]\s+\S+", q) and re.search(
        r"(?im)^\s*[B-D][.)]\s+\S+", q
    ):
        return "mcq"
    if re.search(r"(?i)\b(true|false)\b", q) and re.search(
        r"(?i)\b(statement|claim|assertion|correct)\b", q
    ):
        return "mcq"
    if query_wants_procedure(q):
        return "procedure"
    return "qa"


def mode_prompt_addendum(mode: QueryMode) -> str:
    if mode == "mcq":
        return MCQ_PROMPT_ADDENDUM
    if mode == "summary":
        return SUMMARY_PROMPT_ADDENDUM
    if mode == "qa":
        return QA_PROMPT_ADDENDUM
    return ""


def estimate_confidence(
    *,
    answer: str,
    text_hits: list[dict[str, Any]],
    warnings: list[str],
    mode: str,
) -> str:
    """Return high | medium | low."""
    low_answer = (answer or "").lower()
    if "not found in approved okf bundle" in low_answer:
        return "low"
    if warnings:
        return "medium"
    if not text_hits:
        return "low"
    if mode == "document_faithful":
        return "high"
    # Distance: lower is better for many Milvus metrics (IP inverted elsewhere — use relative)
    try:
        d0 = float(text_hits[0].get("distance") or text_hits[0].get("score") or 0.0)
    except (TypeError, ValueError):
        d0 = 0.0
    if len(text_hits) >= 2:
        try:
            d1 = float(text_hits[1].get("distance") or text_hits[1].get("score") or 0.0)
            gap = abs(d0 - d1)
            if gap < 0.02:
                return "medium"
        except (TypeError, ValueError):
            pass
    if len(str(answer or "").strip()) < 40:
        return "medium"
    return "high"


def suggested_questions(
    query: str,
    text_hits: list[dict[str, Any]],
    *,
    mode: QueryMode | str,
    limit: int = 4,
) -> list[str]:
    """Generate 2–4 follow-up prompts from retrieved titles/sections."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if len(s) < 12 or s.lower() in seen:
            return
        seen.add(s.lower())
        out.append(s)

    if mode == "mcq":
        _add("Explain why the other options are incorrect with citations.")
        _add("Which document page supports the correct option?")
    elif mode == "summary":
        _add("Provide a shorter bullet-point summary of the same topic.")
        _add("What are the key prerequisites mentioned in the source?")
    elif mode == "procedure" or mode == "document_faithful":
        _add("What are the prerequisites for this procedure?")
        _add("Are there any exceptions or special conditions?")
    else:
        _add("Can you explain this in simpler language?")
        _add("Show the step-by-step procedure if available.")

    for h in text_hits[:5]:
        title = str(h.get("title") or "").strip()
        section = str(h.get("section") or "").strip()
        source = str(h.get("source") or "").strip()
        page = h.get("page")
        if title and not title.lower().startswith("ver."):
            _add(f"What does the document say about: {title[:80]}?")
        if section:
            _add(f"Summarize section {section}" + (f" in {source}" if source else ""))
        if source and page:
            _add(f"What else is covered on page {page} of {source}?")
        if len(out) >= limit:
            break

    # Avoid echoing the original query
    qlow = (query or "").strip().lower()
    out = [s for s in out if s.lower() != qlow][:limit]
    return out
