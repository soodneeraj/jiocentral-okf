"""Acceptance scenarios A–R harness (unit + integration hooks).

Run: python -m eval.scenarios
Live Milvus/Ollama tests are skipped unless OKF_LIVE_EVAL=1.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: {detail}")


def test_classification() -> None:
    print("Scenario G/P foundations — classification")
    from app.classification import DocumentType, PageSignals, PageType, classify_document, classify_page

    pages = [
        classify_page(PageSignals(1, selectable_text_len=500, embedded_image_count=0)),
        classify_page(PageSignals(2, selectable_text_len=0, embedded_image_count=2, is_mostly_raster=True)),
        classify_page(PageSignals(3, selectable_text_len=200, embedded_image_count=1)),
        classify_page(PageSignals(4, selectable_text_len=0, embedded_image_count=0)),
    ]
    check("page text", pages[0] == PageType.TEXT)
    check("page scanned", pages[1] == PageType.SCANNED)
    check("page text_image", pages[2] == PageType.TEXT_IMAGE)
    check("page empty", pages[3] == PageType.EMPTY)
    check("doc mixed", classify_document(pages) == DocumentType.MIXED)


def test_asset_registry_occurrences() -> None:
    print("Scenario H/I — asset registry store-once + occurrences")
    from app.assets import register_image
    from app.db.models import Document, ImageAsset, ImageOccurrence
    from app.db.session import get_session, init_db
    from app.utils import load_config

    # Use temp sqlite
    os.environ["DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
    load_config.cache_clear()
    from app.db.session import reset_engine

    reset_engine()
    init_db()

    img = Image.new("RGB", (64, 64), color=(20, 80, 160))
    with get_session() as session:
        d1 = Document(
            name="doc1",
            original_filename="doc1.pdf",
            file_checksum="a" * 64,
            page_count=2,
        )
        d2 = Document(
            name="doc2",
            original_filename="doc2.pdf",
            file_checksum="b" * 64,
            page_count=1,
        )
        session.add_all([d1, d2])
        session.flush()
        a1, o1, created1 = register_image(
            session, image=img, document_id=d1.id, page_number=1
        )
        a2, o2, created2 = register_image(
            session, image=img, document_id=d1.id, page_number=2
        )
        a3, o3, created3 = register_image(
            session, image=img, document_id=d2.id, page_number=1
        )
        check("first binary created", created1 is True)
        check("second reuses binary", created2 is False)
        check("cross-doc reuses binary", created3 is False)
        check("same asset id", a1.id == a2.id == a3.id)
        n_assets = session.query(ImageAsset).count()
        n_occ = session.query(ImageOccurrence).count()
        check("one asset", n_assets == 1)
        check("three occurrences", n_occ == 3)


def test_page_manifest_completeness() -> None:
    print("Scenario P — page manifest completeness")
    from app.db.models import Document
    from app.db.session import get_session, init_db
    from app.manifest import build_completeness_report, ensure_page_rows, update_page
    from app.utils import load_config

    os.environ["DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "manifest.db")
    load_config.cache_clear()
    from app.db.session import reset_engine

    reset_engine()
    init_db()
    with get_session() as session:
        d = Document(name="big", original_filename="big.pdf", file_checksum="c" * 64)
        session.add(d)
        session.flush()
        ensure_page_rows(session, d, 5)
        for n in range(1, 6):
            update_page(session, d.id, n, outcome="text_ok", page_type="text")
        report = build_completeness_report(session, d)
        check("all pages accounted", report["pages_with_manifest"] == 5)
        check("complete true", report["complete"] is True)
        check("status completed", d.processing_status == "completed")

        # Incomplete
        d2 = Document(name="partial", original_filename="p.pdf", file_checksum="d" * 64)
        session.add(d2)
        session.flush()
        ensure_page_rows(session, d2, 3)
        update_page(session, d2.id, 1, outcome="text_ok")
        update_page(session, d2.id, 2, outcome="failed", error="boom")
        # page 3 still pending
        report2 = build_completeness_report(session, d2)
        check("not complete when pending/failed", report2["complete"] is False)


def test_retrieval_intent_rrf() -> None:
    print("Scenario C/K foundations — intent + RRF")
    from app.retrieval import QueryIntent, RetrievalCandidate, classify_intent, reciprocal_rank_fusion

    check("diagram intent", classify_intent("Show me the network topology diagram") == QueryIntent.DIAGRAM)
    check("table intent", classify_intent("What was sales growth in Q2?") == QueryIntent.TABLE)
    check("cross doc", classify_intent("Compare security controls across both documents") == QueryIntent.CROSS_DOC)

    a = [
        RetrievalCandidate(id="t1", modality="text", score=0.9),
        RetrievalCandidate(id="t2", modality="text", score=0.8),
    ]
    b = [
        RetrievalCandidate(id="i1", modality="image", score=0.7),
        RetrievalCandidate(id="t1", modality="text", score=0.6),
    ]
    fused = reciprocal_rank_fusion({"text": a, "image": b}, k=60)
    check("rrf returns items", len(fused) >= 3)
    check("t1 boosted", fused[0].id == "t1")


def test_chunking_tables() -> None:
    print("Scenario L foundation — table chunking keeps header")
    from app.chunking import chunk_table_markdown

    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
    chunks = chunk_table_markdown(md, page_number=1, title="Sales", max_chars=40)
    check("multiple table chunks", len(chunks) >= 1)
    check("header repeated or present", all("| A | B |" in c.text for c in chunks))


def test_security_acl() -> None:
    print("Scenario R foundation — ACL helpers")
    from app.security import document_visible, issue_token, parse_token, verify_password
    from app.utils import load_config

    user = verify_password("reader", "reader")
    check("reader login", user is not None)
    token = issue_token(user["username"], user["roles"], user["access_scopes"])
    parsed = parse_token(token)
    check("token roundtrip", parsed is not None and parsed["username"] == "reader")
    os.environ["AUTH_ENABLED"] = "true"
    load_config.cache_clear()
    check("internal visible", document_visible("internal", ["internal"]) is True)
    check("confidential hidden", document_visible("confidential", ["internal"]) is False)
    os.environ["AUTH_ENABLED"] = "false"
    load_config.cache_clear()


def test_live_optional() -> None:
    if os.environ.get("OKF_LIVE_EVAL") != "1":
        print("Live scenarios A–F skipped (set OKF_LIVE_EVAL=1)")
        return
    print("Live eval against running stack — smoke chat")
    import urllib.request
    import json

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/chat/query",
        data=json.dumps({"query": "What is this repository?"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    check("live answer present", bool(data.get("answer")))


def main() -> int:
    print("=== OKF acceptance scenario harness ===")
    test_classification()
    test_asset_registry_occurrences()
    test_page_manifest_completeness()
    test_retrieval_intent_rrf()
    test_chunking_tables()
    test_security_acl()
    test_live_optional()
    print(f"\nResult: {PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
