"""Local OCR engine abstraction (PaddleOCR with graceful fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.logger import get_logger
from app.utils import load_config

logger = get_logger(__name__)

_paddle_reader: Any = None


@dataclass
class OcrResult:
    text: str
    confidence: float
    lines: list[dict[str, Any]]
    engine: str


def _get_paddle():
    global _paddle_reader
    if _paddle_reader is not None:
        return _paddle_reader
    try:
        from paddleocr import PaddleOCR

        _paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _paddle_reader
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR unavailable: %s", exc)
        _paddle_reader = False
        return None


def ocr_image(image: Image.Image) -> OcrResult:
    """Run local OCR on a PIL image. Never raises; returns empty on failure."""
    cfg = load_config()
    if not cfg.get("ocr_enabled", True):
        return OcrResult(text="", confidence=0.0, lines=[], engine="disabled")

    engine = str(cfg.get("ocr_engine") or "paddleocr")
    if engine == "paddleocr":
        reader = _get_paddle()
        if reader:
            try:
                import numpy as np

                arr = np.array(image.convert("RGB"))
                raw = reader.ocr(arr, cls=True)
                lines: list[dict[str, Any]] = []
                texts: list[str] = []
                confs: list[float] = []
                blocks = raw[0] if raw else []
                for block in blocks or []:
                    try:
                        box, (txt, conf) = block[0], block[1]
                        texts.append(str(txt))
                        confs.append(float(conf))
                        lines.append({"text": txt, "confidence": float(conf), "box": box})
                    except Exception:  # noqa: BLE001
                        continue
                joined = "\n".join(texts).strip()
                avg = sum(confs) / len(confs) if confs else 0.0
                return OcrResult(text=joined, confidence=avg, lines=lines, engine="paddleocr")
            except Exception as exc:  # noqa: BLE001
                logger.warning("PaddleOCR failed: %s", exc)

    # Fallback: empty with flag for review
    return OcrResult(text="", confidence=0.0, lines=[], engine="none")


def ocr_page_image(image: Image.Image) -> OcrResult:
    result = ocr_image(image)
    min_conf = float(load_config().get("ocr_min_confidence", 0.4))
    if result.text and result.confidence < min_conf:
        logger.info(
            "Low-confidence OCR (%.2f < %.2f); retaining text flagged for review",
            result.confidence,
            min_conf,
        )
    return result
