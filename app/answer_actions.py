"""Answer transforms (shorter/detailed/simpler/regenerate) and feedback storage."""

from __future__ import annotations

from typing import Any, Literal

from app.db.models import AnswerFeedback, Message
from app.db.session import get_session, init_db
from app.logger import get_logger
from app.orchestrator import _chat, rag_answer
from app.utils import load_config

logger = get_logger(__name__)

TransformAction = Literal["regenerate", "shorter", "detailed", "simpler"]

_TRANSFORM_INSTRUCTIONS = {
    "shorter": (
        "Rewrite the ANSWER to be shorter and more concise. Keep all factual claims "
        "grounded only in ANSWER and EXCERPTS. Do not add new facts. Preserve citations."
    ),
    "detailed": (
        "Rewrite the ANSWER with more detail and clearer structure, using only facts "
        "present in ANSWER and EXCERPTS. Do not invent information. Preserve citations."
    ),
    "simpler": (
        "Rewrite the ANSWER in simpler language for a non-expert reader. Keep the same "
        "facts from ANSWER and EXCERPTS only. Do not add new facts. Preserve citations."
    ),
}


def transform_answer(
    *,
    action: TransformAction,
    query: str,
    answer: str,
    excerpts: str = "",
    history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    user_id: str = "anonymous",
    top_k: int = 10,
) -> dict[str, Any]:
    """Regenerate via full RAG, or rewrite last answer from excerpts only."""
    cfg = load_config()
    if not cfg.get("answer_actions_enabled", True):
        return {"ok": False, "detail": "Answer actions disabled.", "answer": answer}

    action = action if action in _TRANSFORM_INSTRUCTIONS or action == "regenerate" else "shorter"  # type: ignore[assignment]

    if action == "regenerate":
        result = rag_answer(
            query,
            top_k=top_k,
            history=history,
            conversation_id=conversation_id,
            user_id=user_id,
            skip_cache=True,
        )
        result["ok"] = True
        result["action"] = "regenerate"
        return result

    instruction = _TRANSFORM_INSTRUCTIONS[action]
    system = (
        "You are an enterprise knowledge assistant performing a grounded rewrite.\n"
        f"{instruction}\n"
        "If ANSWER is already a refusal ('Not found in approved OKF bundle'), keep that line.\n"
        "Output only the rewritten answer."
    )
    user = (
        f"ORIGINAL QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"EXCERPTS:\n{excerpts or '(none)'}\n"
    )
    rewritten = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    # Persist as a new assistant message when conversation known
    message_id = None
    if conversation_id and rewritten:
        try:
            init_db()
            with get_session() as session:
                msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=rewritten,
                    evidence={"action": action, "transformed_from_query": query},
                )
                session.add(msg)
                session.flush()
                message_id = msg.id
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transform persist failed: %s", exc)

    return {
        "ok": True,
        "action": action,
        "answer": rewritten or answer,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "mode": action,
        "query_mode": action,
        "confidence": "medium",
        "suggested_questions": [],
        "sources": [],
        "images": [],
        "warnings": [],
    }


def save_feedback(
    *,
    rating: Literal["up", "down"],
    conversation_id: str | None = None,
    message_id: str | None = None,
    query: str = "",
    note: str = "",
    user_id: str = "anonymous",
) -> dict[str, Any]:
    if rating not in {"up", "down"}:
        return {"ok": False, "detail": "rating must be up or down"}
    init_db()
    with get_session() as session:
        row = AnswerFeedback(
            conversation_id=conversation_id or "",
            message_id=message_id or "",
            user_id=user_id,
            query=query[:2000],
            rating=rating,
            note=(note or "")[:2000],
        )
        session.add(row)
        session.flush()
        return {"ok": True, "id": row.id, "rating": rating}
