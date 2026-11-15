"""Unit-level validation from IMPLEMENTATION_PROMPT §16 (no live Milvus/Ollama)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from app.chat import (
    _dedupe_by_file,
    _detect_ambiguous_topics,
    _is_low_information,
    format_citation,
)
from app.ingest import (
    MIN_REFERENCE_STD,
    _dhash,
    _extract_version,
    _grayscale_std,
    _hamming_distance,
)
from app.okf_spec import validate_okf_file


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_validate_okf_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)

        # missing ---
        p = _write(t / "a.md", "no frontmatter here at all with enough body text xx")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for missing ---")
        except ValueError:
            pass

        # <3 parts
        p = _write(t / "b.md", "---\ntype: document\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for <3 parts")
        except ValueError:
            pass

        # invalid YAML
        p = _write(t / "c.md", "---\n: bad: [yaml\n---\n\nbody with more than twenty characters here\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for invalid YAML")
        except ValueError:
            pass

        # non-dict frontmatter
        p = _write(t / "d.md", "---\n- just a list\n---\n\nbody with more than twenty characters here\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for non-dict")
        except ValueError:
            pass

        # missing type
        p = _write(t / "e.md", "---\ntitle: x\n---\n\nbody with more than twenty characters here\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for missing type")
        except ValueError:
            pass

        # invalid type
        p = _write(t / "f.md", "---\ntype: widget\n---\n\nbody with more than twenty characters here\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for invalid type")
        except ValueError:
            pass

        # body ≤20
        p = _write(t / "g.md", "---\ntype: document\n---\n\nshort\n")
        try:
            validate_okf_file(p)
            raise AssertionError("expected ValueError for short body")
        except ValueError:
            pass

        # accepts type + body only
        p = _write(
            t / "h.md",
            "---\ntype: document\n---\n\nbody with more than twenty characters here\n",
        )
        meta, body = validate_okf_file(p)
        assert meta["type"] == "document"
        assert len(body) > 20
    print("OK validate_okf_file")


def test_extract_version() -> None:
    assert _extract_version("Report 16 Dec 25 V1.0.pdf") == "V1.0"
    assert _extract_version("Report 16 Dec 25 v2.pdf") == "V2"
    assert _extract_version("plain_doc.pdf") == ""
    assert _extract_version("V1.0 mid V2.5 end.pdf") == "V2.5"
    print("OK _extract_version")


def test_format_citation() -> None:
    assert format_citation({"source": "a.pdf", "source_loc": "page:1"}) == "a.pdf page:1"
    assert (
        format_citation(
            {
                "source": "a.pdf",
                "source_loc": "page:1",
                "version": "V1.0",
                "section": "4.2",
            }
        )
        == "a.pdf | V1.0 | page:1 | Section 4.2"
    )
    # version only — no empty segment
    assert (
        format_citation({"source": "a.pdf", "source_loc": "page:1", "version": "V1.0"})
        == "a.pdf | V1.0 | page:1"
    )
    # section only
    assert (
        format_citation({"source": "a.pdf", "source_loc": "page:1", "section": "Intro"})
        == "a.pdf | page:1 | Section Intro"
    )
    print("OK format_citation")


def test_is_low_information() -> None:
    assert _is_low_information({"body": "What is the SLA?", "tags": []}) is True
    assert _is_low_information({"body": "x" * 400, "tags": ["section_header"]}) is True
    assert _is_low_information({"body": "x" * 400, "tags": ["title"]}) is True
    assert (
        _is_low_information(
            {"body": "This is a long substantive answer about the SLA process. " * 10, "tags": ["text"]}
        )
        is False
    )
    print("OK _is_low_information")


def test_detect_ambiguous_topics() -> None:
    assert _detect_ambiguous_topics([], 0.03) == []
    assert _detect_ambiguous_topics([{"source": "a", "distance": 0.9}], 0.03) == []

    same = [
        {"source": "a.pdf", "title": "T1", "distance": 0.9},
        {"source": "a.pdf", "title": "T2", "distance": 0.85},
        {"source": "a.pdf", "title": "T3", "distance": 0.8},
    ]
    assert _detect_ambiguous_topics(same, 0.03) == []

    # clear winner (gap > threshold)
    clear = [
        {"source": "a.pdf", "title": "T1", "distance": 0.95},
        {"source": "b.pdf", "title": "T2", "distance": 0.80},
        {"source": "b.pdf", "title": "T3", "distance": 0.70},
    ]
    assert _detect_ambiguous_topics(clear, 0.03) == []

    # competing (gap <= threshold)
    ambiguous = [
        {"source": "a.pdf", "title": "Alpha", "distance": 0.90},
        {"source": "b.pdf", "title": "Beta", "distance": 0.88},
        {"source": "c.pdf", "title": "Gamma", "distance": 0.87},
    ]
    cands = _detect_ambiguous_topics(ambiguous, 0.03)
    assert len(cands) >= 2
    assert {c["source"] for c in cands} >= {"a.pdf", "b.pdf"}
    print("OK _detect_ambiguous_topics")


def test_dedupe_by_file() -> None:
    hits = [
        {"file_path": "concepts/a/x.md", "source_loc": "page:1"},
        {"file_path": "concepts/a/x.md", "source_loc": "page:1"},
        {"file_path": "concepts/a/y.md", "source_loc": "page:1"},
    ]
    out = _dedupe_by_file(hits)
    assert len(out) == 2
    assert out[0]["file_path"] == "concepts/a/x.md"
    assert out[1]["file_path"] == "concepts/a/y.md"
    print("OK _dedupe_by_file")


def test_dhash_and_reference_std() -> None:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    # add a pattern so it's not blank
    for x in range(64):
        for y in range(32):
            img.putpixel((x, y), (200, 200, 200))
    h1 = _dhash(img)
    h2 = _dhash(img.copy())
    assert _hamming_distance(h1, h2) == 0

    blank = Image.new("RGB", (32, 32), color=(10, 10, 10))
    std = _grayscale_std(blank)
    assert std < MIN_REFERENCE_STD
    print("OK _dhash / reference std gate")


def main() -> None:
    test_validate_okf_file()
    test_extract_version()
    test_format_citation()
    test_is_low_information()
    test_detect_ambiguous_topics()
    test_dedupe_by_file()
    test_dhash_and_reference_std()
    print("\nAll unit validations passed.")


if __name__ == "__main__":
    main()
