"""Non-negotiable header / footer / decorative-chrome rejection for image extract.

These rules cannot be disabled by config. Repeated page ornaments, margin
texture strips, and header/footer band banners must never enter the OKF bundle.
"""

from __future__ import annotations

from typing import Any

from PIL import Image


def _gray_stats(img: Image.Image) -> tuple[float, float]:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    if not pixels:
        return 0.0, 0.0
    mean = sum(pixels) / len(pixels)
    var = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return mean, var**0.5


def _band_flags(
    bbox: dict[str, Any] | None,
    page_w: float,
    page_h: float,
    *,
    band_ratio: float,
) -> dict[str, bool | float]:
    if not bbox or page_w <= 0 or page_h <= 0:
        return {
            "in_header": False,
            "in_footer": False,
            "in_band": False,
            "left_margin": False,
            "right_margin": False,
            "width_frac": 0.0,
            "height_frac": 0.0,
        }
    y0 = float(bbox.get("y0") or 0.0)
    y1 = float(bbox.get("y1") or 0.0)
    x0 = float(bbox.get("x0") or 0.0)
    x1 = float(bbox.get("x1") or 0.0)
    top = y0 / page_h
    bot = y1 / page_h
    left = x0 / page_w
    right = x1 / page_w
    in_header = bot <= band_ratio or (top <= band_ratio * 0.85 and (bot - top) <= band_ratio * 1.35)
    in_footer = top >= (1.0 - band_ratio) or (
        bot >= (1.0 - band_ratio * 0.85) and (bot - top) <= band_ratio * 1.35
    )
    return {
        "in_header": in_header,
        "in_footer": in_footer,
        "in_band": in_header or in_footer,
        "left_margin": right <= 0.42,
        "right_margin": left >= 0.58,
        "width_frac": max(0.0, (x1 - x0) / page_w),
        "height_frac": max(0.0, (y1 - y0) / page_h),
    }


def chrome_skip_reason(
    img: Image.Image,
    *,
    bbox: dict[str, Any] | None = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
    band_ratio: float = 0.18,
    is_cross_page_repeat: bool = False,
    is_duplicate_hash: bool = False,
) -> str | None:
    """
    Return a skip reason when the image must be ignored, else None.

    Non-negotiable:
    - cross-page repeats (classic headers/footers/watermarks)
    - duplicate hashes within the same PDF
    - header/footer-band banners and thin ornaments
    - low-entropy vertical margin strips (decorative sidebars)
    - near-empty / spacer plates
    """
    if img is None:
        return "blank"

    w, h = img.size
    if w < 24 or h < 24:
        return "blank"

    if is_cross_page_repeat or is_duplicate_hash:
        return "header_footer"

    aspect = w / max(h, 1)
    mean, std = _gray_stats(img)
    bands = _band_flags(bbox, page_width, page_height, band_ratio=band_ratio)
    in_band = bool(bands["in_band"])
    width_frac = float(bands["width_frac"])
    height_frac = float(bands["height_frac"])
    margin = bool(bands["left_margin"] or bands["right_margin"])

    # Near-empty / washed decorative plates
    if std < 5.0 or mean >= 250.0 or mean <= 5.0:
        return "blank"

    # Wide short banners in header/footer band (grass footers, chapter rails)
    if in_band and aspect >= 3.2:
        return "header_footer"
    if in_band and width_frac >= 0.70 and h <= 320:
        return "header_footer"
    if in_band and (h < 120 or w < 80):
        return "header_footer"

    # Full-bleed header/footer ornaments even when aspect is moderate
    if in_band and width_frac >= 0.85 and height_frac <= 0.28:
        return "header_footer"

    # Extreme wide page dividers anywhere (typical footer graphics)
    if aspect >= 5.5 and h <= 220:
        return "header_footer"

    # Tall / side low-entropy margin strips (cyan side textures)
    if aspect <= 0.40 and std <= 16.0 and (margin or height_frac >= 0.45 or in_band):
        return "header_footer"
    if aspect <= 0.55 and std <= 12.0 and margin and height_frac >= 0.25:
        return "header_footer"
    # Mid-aspect washed panels anchored in a side margin (textbook color washes)
    if margin and std <= 11.0 and mean >= 200.0 and (height_frac >= 0.18 or aspect <= 0.85):
        return "header_footer"

    # Near-white decorative overlays (feathers / ornaments on empty plates)
    gray = img.convert("L")
    pixels = list(gray.getdata())
    white_frac = sum(1 for p in pixels if p >= 248) / max(1, len(pixels))
    if white_frac >= 0.65 and std <= 16.0 and mean >= 235.0:
        return "header_footer"
    if mean >= 240.0 and std <= 14.0 and (in_band or float(bands.get("in_header") or False)):
        return "header_footer"

    return None


def is_non_negotiable_header_footer(
    img: Image.Image,
    **kwargs: Any,
) -> bool:
    reason = chrome_skip_reason(img, **kwargs)
    return reason in {"header_footer", "blank"}
