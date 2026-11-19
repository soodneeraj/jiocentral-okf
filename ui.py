"""ChatGPT-style Gradio UI for Open OKF Enterprise RAG (port 7860).

Single conversation canvas: history sidebar + chat (answers include figures).
"""

from __future__ import annotations

import html
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import gradio as gr
import gradio.networking as _gr_net
import gradio_client.utils as _gr_client_utils

_gr_net.url_ok = lambda _url: True  # type: ignore[assignment]

# Gradio 4.44 can crash api_info when schema additionalProperties is a bool
_orig_json_type = _gr_client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore[no-untyped-def]
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_type(schema, defs)


_gr_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore[assignment]
_gr_client_utils.json_schema_to_python_type = lambda schema: _safe_json_schema_to_python_type(  # type: ignore[assignment]
    schema, schema.get("$defs") if isinstance(schema, dict) else None
)

from app.chat import format_citation, rag_answer
from app.db.models import Conversation, Message
from app.db.session import get_session, init_db
from app.logger import get_logger
from app.utils import images_dir, load_config, okf_bundle_dir

logger = get_logger(__name__)

SOURCE_MARKER = "\n\n---\n**Source:**"
ADMIN_URL = os.environ.get("OKF_ADMIN_URL", "http://127.0.0.1:8000/admin")
API_BASE = os.environ.get("OKF_API_URL", "http://127.0.0.1:8000").rstrip("/")
USER_ID = "anonymous"

_MD_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|webp|gif|bmp|svg))\)",
    re.IGNORECASE,
)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.I)

CUSTOM_CSS = """
:root {
  --okf-bg: #ffffff;
  --okf-side: #171717;
  --okf-side-hover: #212121;
  --okf-side-text: #ececec;
  --okf-side-muted: #8e8ea0;
  --okf-accent: #10a37f;
  --okf-border: #e5e5e5;
  --okf-user: #f4f4f4;
  --okf-composer: #ffffff;
}
.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
  background: var(--okf-bg) !important;
}
footer { display: none !important; }
#okf-root { min-height: 100vh; }
#okf-sidebar {
  background: var(--okf-side) !important;
  color: var(--okf-side-text);
  border-radius: 0 !important;
  min-height: 100vh;
  max-width: 280px;
  padding: 0.85rem 0.75rem !important;
}
#okf-sidebar .brand {
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  margin: 0.15rem 0 0.85rem;
}
#okf-sidebar .brand-sub {
  color: var(--okf-side-muted);
  font-size: 0.75rem;
  margin-top: 1rem;
  line-height: 1.45;
}
#okf-sidebar #okf-new-chat {
  margin-bottom: 0.35rem !important;
}
#okf-sidebar #okf-new-chat button {
  background: transparent !important;
  border: 1px solid #4d4d4f !important;
  color: var(--okf-side-text) !important;
  justify-content: flex-start !important;
  border-radius: 8px !important;
  min-height: 40px !important;
  height: auto !important;
  line-height: 1.3 !important;
  padding: 0.55rem 0.75rem !important;
}
#okf-sidebar .recents-heading {
  color: var(--okf-side-muted);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin: 0.85rem 0.35rem 0.4rem;
}
/* Defeat Gradio .prose / button styles that clip Recents text */
#okf-sidebar #okf-recents-html,
#okf-sidebar #okf-recents-html > .wrap,
#okf-sidebar #okf-recents-html .html-container,
#okf-sidebar #okf-recents-html .prose,
#okf-sidebar #okf-recents-html .prose * {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  overflow: visible !important;
  max-height: none !important;
  height: auto !important;
  line-height: normal !important;
}
#okf-sidebar #okf-recents-html .prose {
  font-size: inherit !important;
  color: inherit !important;
}
.okf-recents-list {
  max-height: calc(100vh - 240px);
  overflow-x: hidden;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-right: 2px;
  background: transparent;
}
.okf-recent-item {
  display: block !important;
  width: 100% !important;
  box-sizing: border-box !important;
  text-align: left !important;
  background: transparent !important;
  color: #ececec !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 12px !important;
  margin: 0 !important;
  font-family: Inter, system-ui, sans-serif !important;
  font-size: 14px !important;
  font-weight: 400 !important;
  line-height: 20px !important;
  min-height: 40px !important;
  height: auto !important;
  max-height: none !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  cursor: pointer !important;
  box-shadow: none !important;
  transform: none !important;
}
.okf-recent-item:hover {
  background: #212121 !important;
}
.okf-recent-item.is-active {
  background: #2f2f2f !important;
}
.okf-recents-empty {
  color: var(--okf-side-muted);
  font-size: 0.82rem;
  padding: 0.75rem 0.5rem;
  text-align: left;
}
#okf-conv-pick {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
  overflow: hidden !important;
}
#okf-sidebar #okf-delete-chat button {
  background: transparent !important;
  border: 1px solid #3f3f46 !important;
  color: #a1a1aa !important;
  border-radius: 8px !important;
  min-height: 36px !important;
  height: auto !important;
  line-height: 1.3 !important;
  margin-top: 0.5rem !important;
}
#okf-main {
  background: var(--okf-bg) !important;
  border-radius: 0 !important;
  min-height: 100vh;
  padding: 0.5rem 0 1rem !important;
  display: flex;
  flex-direction: column;
}
#okf-title {
  text-align: center;
  padding: 1.25rem 1rem 0.5rem;
  max-width: 48rem;
  margin: 0 auto;
}
#okf-title h1, #okf-title h2 {
  font-size: 1.75rem !important;
  font-weight: 600 !important;
  margin: 0.2rem 0 !important;
  letter-spacing: -0.02em;
}
#okf-title p {
  color: #6b6b6b;
  font-size: 0.95rem;
  margin: 0.35rem 0 0 !important;
}
#okf-chatbot {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  max-width: 48rem;
  margin: 0 auto !important;
  width: 100%;
  flex: 1;
}
#okf-chatbot .message-wrap { padding: 0.55rem 0.75rem !important; }
#okf-chatbot .bot,
#okf-chatbot .user {
  border-radius: 18px !important;
}
#okf-chatbot .user {
  background: var(--okf-user) !important;
  border: none !important;
}
#okf-chatbot .bot {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#okf-chatbot img {
  max-width: 100% !important;
  height: auto !important;
  border-radius: 10px !important;
  border: 1px solid var(--okf-border);
  margin: 0.65rem 0;
  display: block;
}
#okf-composer-wrap {
  max-width: 48rem;
  margin: 0.5rem auto 0 !important;
  width: 100%;
  padding: 0 0.75rem;
}
#okf-composer textarea {
  border-radius: 1.5rem !important;
  border: 1px solid var(--okf-border) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
  padding: 0.85rem 1rem !important;
  background: var(--okf-composer) !important;
}
#okf-actions {
  max-width: 48rem;
  margin: 0.35rem auto 0 !important;
  width: 100%;
  padding: 0 0.75rem;
  opacity: 0.9;
}
#okf-status {
  max-width: 48rem;
  margin: 0.25rem auto 0 !important;
  padding: 0 0.75rem;
  color: #6b6b6b;
  font-size: 0.85rem;
}
"""


def _history_to_messages(history: list[list[Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for pair in history or []:
        if not pair or len(pair) < 2:
            continue
        user_msg, bot_msg = pair[0], pair[1]
        if isinstance(user_msg, str) and user_msg.strip():
            messages.append({"role": "user", "content": user_msg})
        if isinstance(bot_msg, str) and bot_msg.strip():
            clean = bot_msg.split(SOURCE_MARKER, 1)[0]
            clean = _MD_IMAGE_RE.sub("", clean)
            clean = _HTML_IMG_RE.sub("", clean)
            clean = re.sub(r"\*_?Mode:.*?_\*", "", clean)
            clean = clean.strip()
            if clean:
                messages.append({"role": "assistant", "content": clean})
    return messages


def _public_image_url(image_path: str) -> str | None:
    rel = (image_path or "").strip().lstrip("./").replace("\\", "/")
    rel = urllib.parse.unquote(rel)
    if not rel:
        return None
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    if "/static/images/" in rel:
        name = urllib.parse.unquote(rel.rsplit("/", 1)[-1])
        return f"{API_BASE}/static/images/{urllib.parse.quote(name)}"
    name = Path(rel).name
    if name and (images_dir() / name).is_file():
        return f"{API_BASE}/static/images/{urllib.parse.quote(name)}"
    if (okf_bundle_dir() / rel).is_file():
        return (
            f"{API_BASE}/static/bundle/"
            f"{'/'.join(urllib.parse.quote(p) for p in rel.split('/'))}"
        )
    return None


def _rewrite_md_images(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        alt = m.group(1) or "figure"
        url = _public_image_url(m.group(2))
        if not url:
            return f"*[{alt} — figure unavailable]*"
        return f"![{alt}]({url})"

    return _MD_IMAGE_RE.sub(repl, text or "")


def _append_extra_images(text: str, images: list[dict[str, Any]] | None) -> str:
    shown = {
        Path(urllib.parse.unquote(m.group(2)).replace("\\", "/")).name
        for m in _MD_IMAGE_RE.finditer(text or "")
    }
    parts = [text or ""]
    for img in images or []:
        ip = str(img.get("image_path") or "")
        name = Path(ip.replace("\\", "/")).name
        if not name or name in shown:
            continue
        url = _public_image_url(ip)
        if not url:
            continue
        label = str(img.get("figure_label") or img.get("caption") or name)
        page = img.get("page")
        caption = f"Page {page} — Fig. {label}" if page is not None else f"Fig. {label}"
        parts.append(f"\n\n![{caption}]({url})\n*{caption}*")
        shown.add(name)
    return "".join(parts).strip()


def _format_assistant_message(
    answer: str,
    *,
    images: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
    mode: str = "",
    confidence: str = "",
    cache_tag: str = "",
) -> str:
    """Full document answer (text + figures) rendered inside the chat bubble."""
    body = _append_extra_images(_rewrite_md_images(answer or ""), images)
    meta_bits: list[str] = []
    if mode:
        meta_bits.append(f"mode: {mode}")
    if confidence:
        meta_bits.append(f"confidence: {confidence}")
    if cache_tag == "hit":
        meta_bits.append("cached")
    head = ""
    if meta_bits:
        head = f"*{' · '.join(meta_bits)}*\n\n"
    if sources:
        src_lines = [f"- {format_citation(s)} [{s.get('type', '')}]" for s in sources]
        body = body.rstrip() + SOURCE_MARKER + "\n" + "\n".join(src_lines)
    return (head + body).strip()


def _list_conversations() -> list[str]:
    """Return encoded rows 'title::conversation_id' for the Recents card list."""
    init_db()
    rows: list[str] = []
    with get_session() as session:
        convs = (
            session.query(Conversation)
            .filter_by(user_id=USER_ID)
            .order_by(Conversation.updated_at.desc())
            .limit(80)
            .all()
        )
        for c in convs:
            title = (c.title or "Untitled").strip() or "Untitled"
            title = title.replace("::", "—")
            if len(title) > 36:
                title = title[:33] + "…"
            rows.append(f"{title}::{c.id}")
    return rows


def _choice_id(choice: str | None) -> str | None:
    if not choice or "::" not in choice:
        return None
    return choice.rsplit("::", 1)[-1]


def _choice_for_id(conversation_id: str | None) -> str | None:
    if not conversation_id:
        return None
    for row in _list_conversations():
        if row.endswith(f"::{conversation_id}"):
            return row
    return None


def _render_recents_html(selected_id: str | None = None) -> str:
    """Simple ChatGPT-style list rows (one entry per chat)."""
    rows = _list_conversations()
    if not rows:
        return (
            '<div class="okf-recents-list">'
            '<div class="okf-recents-empty">No chats yet</div>'
            "</div>"
        )
    items: list[str] = []
    for row in rows:
        title, cid = row.rsplit("::", 1)
        active = " is-active" if selected_id and cid == selected_id else ""
        items.append(
            f'<div class="okf-recent-item{active}" role="button" tabindex="0" '
            f'data-choice="{html.escape(row, quote=True)}">'
            f"{html.escape(title)}</div>"
        )
    return f'<div class="okf-recents-list">{"".join(items)}</div>'


def _refresh_history_ui(selected_id: str | None = None) -> tuple[Any, Any]:
    value = _choice_for_id(selected_id)
    return (
        gr.update(value=_render_recents_html(selected_id)),
        gr.update(value=value),
    )


RECENTS_JS = """
() => {
  if (window.__okfRecentsWired) return;
  window.__okfRecentsWired = true;
  const pick = (choice) => {
    const root = document.querySelector('#okf-conv-pick');
    if (!root) return;
    const input = root.querySelector('textarea') || root.querySelector('input');
    if (!input) return;
    const proto = input.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(input, choice);
    else input.value = choice;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  document.addEventListener('click', (ev) => {
    const item = ev.target && ev.target.closest && ev.target.closest('.okf-recent-item');
    if (!item) return;
    pick(item.getAttribute('data-choice') || '');
  });
}
"""


def _pack_result(
    history: list[list[Any]],
    conversation_id: str | None,
    result: dict[str, Any],
    *,
    query: str,
) -> tuple[Any, ...]:
    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    images = result.get("images") or []
    cache_tag = result.get("cache") or "miss"
    new_id = result.get("conversation_id") or conversation_id
    mode = str(result.get("query_mode") or result.get("mode") or "qa")
    confidence = str(result.get("confidence") or "")
    message_id = result.get("message_id")
    excerpts = ""
    if sources:
        excerpts = "\n\n".join(
            f"[{format_citation(s)}]\n{str(s.get('body') or '')[:1500]}" for s in sources[:2]
        )
    for ev in (result.get("evidence_map") or [])[:2]:
        if not excerpts:
            excerpts = str(ev.get("citation") or "")

    chat_text = _format_assistant_message(
        answer,
        images=images,
        sources=sources,
        mode=mode,
        confidence=confidence,
        cache_tag=cache_tag,
    )

    history = [list(row) for row in history]
    if history and history[-1] and history[-1][0] == query:
        history[-1][1] = chat_text
    else:
        history.append([query, chat_text])

    return (
        history,
        new_id,
        "",
        *_refresh_history_ui(new_id),
        query,
        answer,
        excerpts,
        message_id,
        "",
    )


def new_chat():
    return (
        [],
        None,
        "",
        *_refresh_history_ui(None),
        "",
        "",
        "",
        None,
        "",
    )


def load_conversation(choice: str | None):
    conversation_id = _choice_id(choice)
    empty = (
        [],
        None,
        "",
        "",
        "",
        None,
        "",
        gr.update(value=_render_recents_html(None)),
    )
    if not conversation_id:
        return empty
    init_db()
    history: list[list[Any]] = []
    last_query = ""
    last_answer = ""
    message_id = None
    with get_session() as session:
        c = session.get(Conversation, conversation_id)
        if not c:
            return empty
        msgs = (
            session.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        pending_user: str | None = None
        for m in msgs:
            if m.role == "user":
                if pending_user is not None:
                    history.append([pending_user, ""])
                pending_user = m.content or ""
                last_query = pending_user
            elif m.role == "assistant":
                content = m.content or ""
                evidence = m.evidence or {}
                images = evidence.get("images") or []
                sources = evidence.get("sources") or []
                mode = str(evidence.get("query_mode") or evidence.get("mode") or "")
                conf = str(evidence.get("confidence") or "")
                chat_text = _format_assistant_message(
                    content,
                    images=images,
                    sources=sources,
                    mode=mode,
                    confidence=conf,
                )
                if pending_user is not None:
                    history.append([pending_user, chat_text])
                    pending_user = None
                else:
                    history.append(["", chat_text])
                last_answer = content
                message_id = m.id
        if pending_user is not None:
            history.append([pending_user, ""])
    return (
        history,
        conversation_id,
        last_query,
        last_answer,
        "",
        message_id,
        "",
        gr.update(value=_render_recents_html(conversation_id)),
    )


def delete_conversation(conversation_id: str | None):
    if conversation_id:
        init_db()
        with get_session() as session:
            c = session.get(Conversation, conversation_id)
            if c:
                session.delete(c)
    return new_chat()


def respond(
    message: str,
    history: list[list[Any]] | None,
    conversation_id: str | None,
):
    history = [list(row) for row in (history or [])]
    message = (message or "").strip()
    if not message:
        return (
            history,
            conversation_id,
            "",
            *_refresh_history_ui(conversation_id),
            "",
            "",
            "",
            None,
            "",
        )

    cfg = load_config()
    history_turns = int(cfg.get("chat_history_turns", 6))
    max_pairs = max(1, history_turns // 2)
    prior = _history_to_messages(history[-max_pairs:]) if history else []

    try:
        result = rag_answer(
            message,
            top_k=int(cfg.get("top_k", 10)),
            history=prior or None,
            conversation_id=conversation_id,
            user_id=USER_ID,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        err = f"Error: {exc}"
        history.append([message, err])
        return (
            history,
            conversation_id,
            "",
            *_refresh_history_ui(conversation_id),
            message,
            err,
            "",
            None,
            err,
        )

    return _pack_result(history, conversation_id, result, query=message)


def apply_transform(
    action: str,
    history: list[list[Any]] | None,
    conversation_id: str | None,
    last_query: str,
    last_answer: str,
    last_excerpts: str,
):
    from app.answer_actions import transform_answer

    history = [list(row) for row in (history or [])]
    q = (last_query or "").strip()
    if not q:
        return (
            history,
            conversation_id,
            "",
            *_refresh_history_ui(conversation_id),
            last_query,
            last_answer,
            last_excerpts,
            None,
            "Ask a question first.",
        )
    cfg = load_config()
    prior = _history_to_messages(history[-3:]) if history else []
    result = transform_answer(
        action=action,  # type: ignore[arg-type]
        query=q,
        answer=last_answer or "",
        excerpts=last_excerpts or "",
        history=prior or None,
        conversation_id=conversation_id,
        user_id=USER_ID,
        top_k=int(cfg.get("top_k", 10)),
    )
    if not result.get("ok"):
        return (
            history,
            conversation_id,
            "",
            *_refresh_history_ui(conversation_id),
            last_query,
            last_answer,
            last_excerpts,
            None,
            str(result.get("detail") or "Action failed"),
        )
    if action != "regenerate":
        result.setdefault("images", [])
        result.setdefault("sources", [])
        result.setdefault("query_mode", action)
        result.setdefault("confidence", "medium")
    return _pack_result(history, conversation_id, result, query=q)


def submit_feedback(
    rating: str,
    conversation_id: str | None,
    message_id: str | None,
    last_query: str,
):
    from app.answer_actions import save_feedback

    res = save_feedback(
        rating=rating,  # type: ignore[arg-type]
        conversation_id=conversation_id,
        message_id=message_id,
        query=last_query or "",
        user_id=USER_ID,
    )
    label = "Thanks for the feedback." if res.get("ok") else str(res.get("detail") or "Failed")
    return label


def build_demo() -> gr.Blocks:
    cfg = load_config()
    with gr.Blocks(
        title="OKF Chat",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
        js=RECENTS_JS,
        elem_id="okf-root",
    ) as demo:
        conversation_id = gr.State(None)
        last_query = gr.State("")
        last_answer = gr.State("")
        last_excerpts = gr.State("")
        message_id = gr.State(None)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=240, elem_id="okf-sidebar"):
                gr.HTML("<div class='brand'>OKF Chat</div>")
                new_btn = gr.Button(
                    "+ New chat",
                    variant="secondary",
                    elem_id="okf-new-chat",
                )
                gr.HTML("<div class='recents-heading'>Recents</div>")
                recents_html = gr.HTML(
                    value=_render_recents_html(),
                    elem_id="okf-recents-html",
                )
                conv_pick = gr.Textbox(
                    value="",
                    label="conv",
                    show_label=False,
                    elem_id="okf-conv-pick",
                    lines=1,
                )
                delete_btn = gr.Button(
                    "Delete chat",
                    variant="secondary",
                    size="sm",
                    elem_id="okf-delete-chat",
                )
                gr.HTML(
                    f"<div class='brand-sub'>"
                    f"Model: {html.escape(str(cfg.get('chat_model') or ''))}<br/>"
                    f"<a href='{html.escape(ADMIN_URL)}' target='_blank' "
                    f"rel='noopener' style='color:#10a37f;'>Admin</a>"
                    f"</div>"
                )

            with gr.Column(scale=8, elem_id="okf-main"):
                gr.Markdown(
                    "## How can I help?\n"
                    "Ask questions, MCQs, summaries, or procedures from your OKF docs.",
                    elem_id="okf-title",
                )
                chatbot = gr.Chatbot(
                    label="",
                    height=560,
                    type="tuples",
                    render_markdown=True,
                    bubble_full_width=False,
                    show_label=False,
                    avatar_images=(None, None),
                    elem_id="okf-chatbot",
                )
                with gr.Row(elem_id="okf-composer-wrap"):
                    with gr.Row(elem_id="okf-composer"):
                        msg = gr.Textbox(
                            placeholder="Ask OKF…",
                            show_label=False,
                            lines=1,
                            max_lines=6,
                            scale=10,
                            container=False,
                            autofocus=True,
                        )
                        send = gr.Button("Send", variant="primary", scale=1, min_width=88)

                with gr.Row(elem_id="okf-actions"):
                    btn_regen = gr.Button("Regenerate", size="sm")
                    btn_short = gr.Button("Shorter", size="sm")
                    btn_detail = gr.Button("More detail", size="sm")
                    btn_simple = gr.Button("Simpler", size="sm")
                    btn_up = gr.Button("👍", size="sm")
                    btn_down = gr.Button("👎", size="sm")

                status_msg = gr.Markdown("", elem_id="okf-status")

        # chatbot, conversation_id, msg, recents_html, conv_pick,
        # last_query, last_answer, last_excerpts, message_id, status
        chat_outputs = [
            chatbot,
            conversation_id,
            msg,
            recents_html,
            conv_pick,
            last_query,
            last_answer,
            last_excerpts,
            message_id,
            status_msg,
        ]

        new_btn.click(new_chat, outputs=chat_outputs, api_name=False)
        delete_btn.click(
            delete_conversation,
            inputs=[conversation_id],
            outputs=chat_outputs,
            api_name=False,
        )
        conv_pick.change(
            load_conversation,
            inputs=[conv_pick],
            outputs=[
                chatbot,
                conversation_id,
                last_query,
                last_answer,
                last_excerpts,
                message_id,
                status_msg,
                recents_html,
            ],
            api_name=False,
        )
        send.click(
            respond,
            inputs=[msg, chatbot, conversation_id],
            outputs=chat_outputs,
            api_name=False,
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, conversation_id],
            outputs=chat_outputs,
            api_name=False,
        )

        for action, btn in (
            ("regenerate", btn_regen),
            ("shorter", btn_short),
            ("detailed", btn_detail),
            ("simpler", btn_simple),
        ):
            btn.click(
                lambda hist, cid, lq, la, le, act=action: apply_transform(
                    act, hist, cid, lq, la, le
                ),
                inputs=[chatbot, conversation_id, last_query, last_answer, last_excerpts],
                outputs=chat_outputs,
                api_name=False,
            )

        btn_up.click(
            lambda cid, mid, lq: submit_feedback("up", cid, mid, lq),
            inputs=[conversation_id, message_id, last_query],
            outputs=[status_msg],
            api_name=False,
        )
        btn_down.click(
            lambda cid, mid, lq: submit_feedback("down", cid, mid, lq),
            inputs=[conversation_id, message_id, last_query],
            outputs=[status_msg],
            api_name=False,
        )
    return demo


if __name__ == "__main__":
    print(f"[okf-ui] starting Gradio on 127.0.0.1:7860 ; admin={ADMIN_URL}", flush=True)
    print(f"[okf-ui] document images via {API_BASE}/static/images/", flush=True)
    init_db()
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(images_dir()), str(okf_bundle_dir())],
        share=False,
        inbrowser=False,
        show_api=False,
    )
