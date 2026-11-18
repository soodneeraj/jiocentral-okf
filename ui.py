"""Gradio Chat UI for Open OKF Enterprise RAG (port 7860).

Chatbot shows the dialogue text. A Document view (HTML) renders the latest
answer with steps and figures via FastAPI /static/images URLs — Gradio’s
native file-tuple chatbot breaks page load on 4.44 / Windows.
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

_gr_net.url_ok = lambda _url: True  # type: ignore[assignment]

from app.chat import format_citation, rag_answer
from app.logger import get_logger
from app.utils import images_dir, load_config, okf_bundle_dir

logger = get_logger(__name__)

SOURCE_MARKER = "\n\n---\n**Source:**"
ADMIN_URL = os.environ.get("OKF_ADMIN_URL", "http://127.0.0.1:8000/admin")
API_BASE = os.environ.get("OKF_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Paths may contain parentheses (e.g. eOffice_(eFile)_...png) — do not stop at first ')'
_MD_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|webp|gif|bmp|svg))\)",
    re.IGNORECASE,
)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.I)


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
            clean = clean.strip()
            if clean:
                messages.append({"role": "assistant", "content": clean})
    return messages


def _public_image_url(image_path: str) -> str | None:
    rel = (image_path or "").strip().lstrip("./").replace("\\", "/")
    # Undo markdown-safe encoding from faithful answers
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
    # bundle fallback
    if (okf_bundle_dir() / rel).is_file():
        return f"{API_BASE}/static/bundle/{'/'.join(urllib.parse.quote(p) for p in rel.split('/'))}"
    return None


def _plain_chat_text(answer: str) -> str:
    """Chatbot bubble: keep step text, drop raw image markdown (shown in HTML view)."""
    text = _MD_IMAGE_RE.sub("", answer or "")
    text = _HTML_IMG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _answer_to_html(answer: str, images: list[dict[str, Any]] | None = None) -> str:
    """Build HTML document view with steps + visible figures."""
    raw = answer or ""
    parts: list[str] = [
        '<div style="font-family:Segoe UI,system-ui,sans-serif;line-height:1.45;'
        'max-width:920px;">'
    ]
    pos = 0
    shown: set[str] = set()
    for m in _MD_IMAGE_RE.finditer(raw):
        if m.start() > pos:
            chunk = raw[pos : m.start()]
            parts.append(
                f"<div style='white-space:pre-wrap;margin:0.5rem 0;'>"
                f"{html.escape(chunk).replace(chr(10), '<br/>')}</div>"
            )
        alt = m.group(1) or "figure"
        url = _public_image_url(m.group(2))
        if url:
            shown.add(Path(m.group(2).replace('\\', '/')).name)
            parts.append(
                f'<figure style="margin:0.75rem 0;">'
                f'<img src="{html.escape(url)}" alt="{html.escape(alt)}" '
                f'style="max-width:100%;height:auto;border:1px solid #ccc;'
                f'border-radius:6px;display:block;background:#fafafa;" />'
                f'<figcaption style="color:#555;font-size:0.9rem;margin-top:0.35rem;">'
                f"{html.escape(alt)}</figcaption></figure>"
            )
        else:
            parts.append(
                f"<p style='color:#a00;'>[figure missing: {html.escape(m.group(2))}]</p>"
            )
        pos = m.end()
    tail = raw[pos:]
    if tail.strip():
        parts.append(
            f"<div style='white-space:pre-wrap;margin:0.5rem 0;'>"
            f"{html.escape(tail).replace(chr(10), '<br/>')}</div>"
        )

    # Append any retrieved images not already embedded
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
        parts.append(
            f'<figure style="margin:0.75rem 0;">'
            f'<img src="{html.escape(url)}" alt="{html.escape(label)}" '
            f'style="max-width:100%;height:auto;border:1px solid #ccc;'
            f'border-radius:6px;display:block;" />'
            f'<figcaption style="color:#555;font-size:0.9rem;margin-top:0.35rem;">'
            f"Page {html.escape(str(page))} — Fig. {html.escape(label)}"
            f"</figcaption></figure>"
        )
        shown.add(name)

    if len(parts) == 1:
        parts.append("<p><em>No answer content.</em></p>")
    parts.append("</div>")
    return "".join(parts)


def respond(
    message: str,
    history: list[list[Any]] | None,
) -> tuple[list[list[Any]], str, str]:
    history = [list(row) for row in (history or [])]
    message = (message or "").strip()
    if not message:
        return history, "", ""

    cfg = load_config()
    history_turns = int(cfg.get("chat_history_turns", 6))
    max_pairs = max(1, history_turns // 2)
    prior = _history_to_messages(history[-max_pairs:]) if history else []

    try:
        result = rag_answer(
            message,
            top_k=int(cfg.get("top_k", 10)),
            history=prior or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        err = f"Error: {exc}"
        history.append([message, err])
        return history, "", f"<pre>{html.escape(err)}</pre>"

    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    images = result.get("images") or []
    cache_tag = result.get("cache") or "miss"

    chat_text = _plain_chat_text(answer)
    if cache_tag == "hit":
        chat_text += "\n\n_⚡ served from Elasticsearch cache_"
    if sources:
        src_lines = [f"- {format_citation(s)} [{s.get('type', '')}]" for s in sources]
        chat_text += SOURCE_MARKER + "\n" + "\n".join(src_lines)
    # Note in chat that figures are in the panel below
    n_md = len(_MD_IMAGE_RE.findall(answer))
    n_img = max(n_md, len(images or []))
    if n_img:
        chat_text += f"\n\n_📷 {n_img} figure(s) shown in Document view below._"

    history.append([message, chat_text])
    doc_html = _answer_to_html(answer, images)
    logger.info(
        "chat ok mode=%s md_images=%d payload_images=%d",
        result.get("mode"),
        n_md,
        len(images or []),
    )
    return history, "", doc_html


def build_demo() -> gr.Blocks:
    cfg = load_config()
    with gr.Blocks(title="Open OKF Enterprise RAG", theme=gr.themes.Default()) as demo:
        gr.Markdown(
            f"# Open OKF Enterprise RAG\n"
            f"Model: `{cfg.get('chat_model')}` · "
            f"[Admin PDF upload]({ADMIN_URL})"
        )
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    type="tuples",
                    render_markdown=True,
                    bubble_full_width=True,
                )
                msg = gr.Textbox(
                    placeholder="Ask about procedures, steps, figures…",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=1):
                gr.Markdown("### Document view (steps + images)")
                doc_view = gr.HTML(
                    value="<p style='color:#666;'>Figures for the latest answer appear here.</p>"
                )

        send.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, doc_view],
            api_name=False,
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, doc_view],
            api_name=False,
        )
        clear.click(
            lambda: ([], "", "<p style='color:#666;'>Figures for the latest answer appear here.</p>"),
            outputs=[chatbot, msg, doc_view],
            api_name=False,
        )
    return demo


if __name__ == "__main__":
    print(f"[okf-ui] starting Gradio on 127.0.0.1:7860 ; admin={ADMIN_URL}", flush=True)
    print(f"[okf-ui] document images via {API_BASE}/static/images/", flush=True)
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(images_dir()), str(okf_bundle_dir())],
        share=False,
        inbrowser=False,
        show_api=False,
    )
