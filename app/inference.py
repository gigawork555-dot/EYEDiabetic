from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from app.model_loader import get_model_and_meta
from app.utils.image_utils import MAX_BYTES, allowed_image, load_rgb_224_from_bytes, sanitize_filename

logger = logging.getLogger("app.inference")

THRESHOLD = 0.5


def _round4(x: float) -> float:
    return float(f"{float(x):.4f}")


def _utc_iso() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def predict_image_bytes(image_bytes: bytes, original_name: str) -> Dict[str, Any]:
    filename = sanitize_filename(original_name)

    if not image_bytes:
        raise ValueError("ไฟล์ว่างเปล่า")
    if len(image_bytes) > MAX_BYTES:
        raise ValueError("ไฟล์มีขนาดเกิน 5MB")
    if not allowed_image(filename, image_bytes):
        raise ValueError("ชนิดไฟล์ไม่ถูกต้อง (รองรับเฉพาะ JPG/PNG)")

    x = load_rgb_224_from_bytes(image_bytes)  # (224,224,3) float32 0..255
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)

    model, _ = get_model_and_meta()
    y = model.predict(x, verbose=0)

    prob_no = float(np.squeeze(y))
    if np.isnan(prob_no) or np.isinf(prob_no):
        raise RuntimeError("โมเดลให้ผลลัพธ์ไม่ถูกต้อง")
    prob_no = float(np.clip(prob_no, 0.0, 1.0))
    prob_dr = 1.0 - prob_no

    if prob_dr >= THRESHOLD:
        pred = "DR"
        confidence = prob_dr
    else:
        pred = "No DR"
        confidence = prob_no

    result = {
        "filename": filename,
        "prediction": pred,
        "confidence": _round4(confidence),
        "probabilities": {"No DR": _round4(prob_no), "DR": _round4(prob_dr)},
    }

    logger.info(
        "ts=%s filename=%s pred=%s confidence=%.4f prob_dr=%.4f",
        _utc_iso(),
        filename,
        pred,
        float(result["confidence"]),
        float(result["probabilities"]["DR"]),
    )

    return result


def get_system_info() -> Dict[str, Any]:
    try:
        _, meta = get_model_and_meta()
        return {
            "status": "ok",
            "model_version": meta.model_version,
            "last_updated": meta.last_updated,
            "threshold": THRESHOLD,
        }
    except Exception as e:
        logger.exception("System info error")
        return {
            "status": "error",
            "error": str(e),
            "model_version": None,
            "last_updated": None,
            "threshold": THRESHOLD,
        }

