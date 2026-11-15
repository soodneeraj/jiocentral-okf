"""Gradio Chat UI for Open OKF Enterprise RAG (port 7860).

Admin PDF upload is on the FastAPI backend: http://127.0.0.1:8000/admin
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

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


def _history_to_messages(history: list[tuple[str | None, str | None]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            clean = bot_msg.split(SOURCE_MARKER, 1)[0]
            messages.append({"role": "assistant", "content": clean})
    return messages


def _local_image_url(image_path: str) -> str:
    abs_path = (okf_bundle_dir() / image_path).resolve()
    if abs_path.drive:
        rest = "/".join(
            urllib.parse.quote(p, safe="") for p in abs_path.parts[1:]
        )
        encoded = f"{abs_path.drive}/{rest}".replace("\\", "/")
    else:
        encoded = "/".join(
            urllib.parse.quote(p, safe="") for p in abs_path.parts
        )
    return f"/file={encoded}"


def chat_fn(message: str, history: list[tuple[str | None, str | None]]) -> str:
    cfg = load_config()
    history_turns = int(cfg.get("chat_history_turns", 6))
    max_pairs = max(1, history_turns // 2)
    trimmed_history = history[-max_pairs:] if history else []
    messages = _history_to_messages(trimmed_history)

    result = rag_answer(message, top_k=int(cfg.get("top_k", 10)), history=messages or None)
    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    images = result.get("images") or []
    cache_tag = result.get("cache") or "miss"

    evidence_map = result.get("evidence_map") or []

    parts = [answer]
    if cache_tag == "hit":
        parts.append("\n\n_⚡ served from Elasticsearch cache_")
    if sources:
        src_lines = [
            f"- {format_citation(s)} [{s.get('type', '')}]" for s in sources
        ]
        parts.append(SOURCE_MARKER + "\n" + "\n".join(src_lines))

    if evidence_map:
        ev_lines = ["\n\n**Evidence map (text ↔ page images):**"]
        for ev in evidence_map:
            ev_lines.append(
                f"- `{ev.get('citation')}` — {ev.get('title') or ''}"
            )
            for img in ev.get("images") or []:
                ip = str(img.get("image_path") or "")
                if not ip:
                    continue
                url = _local_image_url(ip)
                ev_lines.append(
                    f"  - page {img.get('page')}: ![{Path(ip).name}]({url})"
                )
                if img.get("caption"):
                    ev_lines.append(f"    *{img.get('caption')}*")
        parts.append("\n".join(ev_lines))
    elif images:
        img_blocks = ["\n\n**Related images (page-mapped):**"]
        for img in images:
            ip = str(img.get("image_path") or "")
            caption = str(img.get("caption") or "")
            page = img.get("page")
            mapped = img.get("mapped_from_page")
            url = _local_image_url(ip)
            img_blocks.append(
                f"- source page `{page}` (from text page `{mapped}`): "
                f"![{Path(ip).name}]({url})"
            )
            if caption:
                img_blocks.append(f"  *{caption}*")
        parts.append("\n".join(img_blocks))

    return "\n".join(parts)


def build_demo() -> gr.ChatInterface:
    cfg = load_config()
    return gr.ChatInterface(
        fn=chat_fn,
        title="Open OKF Enterprise RAG",
        description=(
            f"Ask questions against the approved OKF knowledge bundle "
            f"(`{cfg.get('chat_model')}`).\n\n"
            f"**Admin — upload PDFs into RAG:** {ADMIN_URL}"
        ),
    )


if __name__ == "__main__":
    print(f"[okf-ui] starting Gradio on 127.0.0.1:7860 ; admin={ADMIN_URL}", flush=True)
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(images_dir())],
        share=False,
        inbrowser=False,
        show_api=False,
    )
