"""Global image asset registry: store once, retain every occurrence."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy.orm import Session

from app.db.models import ImageAsset, ImageOccurrence
from app.logger import get_logger
from app.utils import assets_registry_dir, bytes_sha256, load_config

logger = get_logger(__name__)


def pixmap_to_pil(pix: Any) -> Image.Image:
    """
    Convert a PyMuPDF Pixmap to an RGB PIL image without color loss.

    Textbook PDFs commonly embed:
    - DeviceCMYK / Indexed→CMYK (JPX) — must not be read as RGBA
    - DeviceN/Separation **Black** (single CMYK K channel) — samples are *ink
      coverage* (high = dark), not luminance. Reading them as DeviceGray yields
      photographic negatives (author portraits in English.pdf).

    Always normalize non-RGB spaces through MuPDF's DeviceRGB transform, and
    keep the pixmap's native width/height (document embedded resolution).
    """
    import fitz  # PyMuPDF

    if pix is None:
        raise ValueError("pixmap is None")

    working = pix
    cs = working.colorspace
    cs_name = (cs.name if cs is not None else "") or ""
    cs_n = int(cs.n) if cs is not None else int(working.n) - int(bool(working.alpha))
    low = cs_name.lower()

    already_rgb = (
        cs is not None
        and cs_n == 3
        and int(working.n) == 3
        and not working.alpha
        and "rgb" in low
        and "devicen" not in low
        and "cmyk" not in low
    )

    if not already_rgb:
        # CMYK, DeviceN/Separation (Black ink), Gray, Indexed, ICC, alpha → RGB
        try:
            working = fitz.Pixmap(fitz.csRGB, working)
        except Exception:
            # Rare: colorspace None / mask-only — fall back to raw samples
            if int(working.n) - int(bool(working.alpha)) == 1:
                if working.alpha:
                    working = fitz.Pixmap(working, 0)
                img = Image.frombytes(
                    "L", (working.width, working.height), working.samples
                )
                return img.convert("RGB")
            raise

    if working.alpha:
        working = fitz.Pixmap(working, 0)
    if working.n != 3:
        working = fitz.Pixmap(fitz.csRGB, working)

    return Image.frombytes("RGB", (working.width, working.height), working.samples)


def _dhash_hex(img: Image.Image) -> str:
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def _hamming(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 64


def register_image(
    session: Session,
    *,
    image: Image.Image,
    document_id: str,
    page_number: int,
    image_role: str = "content",
    indexable: bool = True,
    caption: str = "",
    bbox: dict[str, Any] | None = None,
    section_heading: str = "",
    ocr_text: str = "",
    description: str = "",
) -> tuple[ImageAsset, ImageOccurrence, bool]:
    """Persist binary once by SHA-256; always create an occurrence row.

    Returns ``(asset, occurrence, created_new_binary)``.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    content_hash = bytes_sha256(raw)
    phash = _dhash_hex(image)

    asset = session.query(ImageAsset).filter_by(content_hash=content_hash).one_or_none()
    created = False
    if asset is None:
        abs_path = assets_registry_dir() / content_hash[:2] / f"{content_hash}.png"
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            abs_path.write_bytes(raw)
        asset = ImageAsset(
            content_hash=content_hash,
            perceptual_hash=phash,
            image_path=f"assets/registry/{content_hash[:2]}/{content_hash}.png",
            width=image.width,
            height=image.height,
            image_format="png",
            image_type=image_role,
            image_description=description,
            ocr_text=ocr_text,
            storage_status="stored",
        )
        session.add(asset)
        session.flush()
        created = True
    else:
        # Enrich description/OCR if previously empty
        if description and not asset.image_description:
            asset.image_description = description
        if ocr_text and not asset.ocr_text:
            asset.ocr_text = ocr_text

    occ = ImageOccurrence(
        asset_id=asset.id,
        document_id=document_id,
        page_number=page_number,
        bbox=bbox or {},
        image_role=image_role,
        indexable=indexable,
        caption=caption,
        section_heading=section_heading,
        associated_chunk_ids=[],
    )
    session.add(occ)
    session.flush()
    return asset, occ, created


def find_perceptual_matches(
    session: Session,
    phash: str,
    *,
    max_hamming: int | None = None,
) -> list[ImageAsset]:
    cfg = load_config()
    threshold = int(max_hamming if max_hamming is not None else cfg.get("brand_image_max_hamming", 10))
    matches: list[ImageAsset] = []
    for asset in session.query(ImageAsset).all():
        if asset.perceptual_hash and _hamming(phash, asset.perceptual_hash) <= threshold:
            matches.append(asset)
    return matches


def absolute_asset_path(asset: ImageAsset) -> Path:
    from app.utils import okf_bundle_dir

    return okf_bundle_dir() / asset.image_path


def occurrence_payload(occ: ImageOccurrence, asset: ImageAsset | None = None) -> dict[str, Any]:
    a = asset or occ.asset
    return {
        "occurrence_id": occ.id,
        "global_image_asset_id": a.id if a else occ.asset_id,
        "document_id": occ.document_id,
        "page_number": occ.page_number,
        "image_path": a.image_path if a else "",
        "content_hash": a.content_hash if a else "",
        "perceptual_hash": a.perceptual_hash if a else "",
        "image_role": occ.image_role,
        "indexable": occ.indexable,
        "caption": occ.caption,
        "image_description": a.image_description if a else "",
        "ocr_text": a.ocr_text if a else "",
        "associated_chunk_ids": occ.associated_chunk_ids or [],
        "bbox": occ.bbox or {},
    }
