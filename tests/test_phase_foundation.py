"""Unit tests for classification, chunking, RRF, security."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.classification import PageSignals, PageType, classify_page
from app.chunking import pack_paragraphs
from app.retrieval import QueryIntent, classify_intent
from app.security import issue_token, parse_token


def test_classify_page_text():
    assert classify_page(PageSignals(1, selectable_text_len=200)) == PageType.TEXT


def test_pack_paragraphs():
    chunks = pack_paragraphs(["hello world " * 20, "next para"], page_number=1, max_chars=80)
    assert len(chunks) >= 1
    assert chunks[0].content_hash


def test_intent():
    assert classify_intent("show the architecture diagram") == QueryIntent.DIAGRAM


def test_token():
    tok = issue_token("u", ["reader"], ["internal"])
    assert parse_token(tok)["username"] == "u"
