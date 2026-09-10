from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    base = Path(name or "").name
    base = _SAFE_RE.sub("_", base)
    base = base.strip("._-")
    if not base:
        base = "image"
    return base[:200]


def allowed_image(filename: str, image_bytes: bytes) -> bool:
    try:
        ext = Path(filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            return False
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
            fmt = (img.format or "").upper()
        return fmt in {"JPEG", "PNG"}
    except Exception:
        return False


def load_rgb_224_from_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            resample = getattr(Image, "Resampling", Image).BILINEAR
            img = img.resize((224, 224), resample=resample)
            arr = np.asarray(img, dtype=np.float32)
        return arr
    except UnidentifiedImageError as e:
        raise ValueError("ไฟล์รูปอาจเสียหาย หรือไม่ใช่ไฟล์รูปที่รองรับ") from e
