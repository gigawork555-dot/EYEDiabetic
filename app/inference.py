from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.model_loader import DEFAULT_MODEL_KEY, get_all_meta, get_model_and_meta, list_available_models
from app.models.registry import MODEL_REGISTRY
from app.utils.eval_utils import compute_metrics
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


def _prepare_input(image_bytes: bytes, original_name: str) -> Tuple[np.ndarray, str]:
    filename = sanitize_filename(original_name)

    if not image_bytes:
        raise ValueError("ไฟล์ว่างเปล่า")
    if len(image_bytes) > MAX_BYTES:
        raise ValueError("ไฟล์มีขนาดเกิน 5MB")
    if not allowed_image(filename, image_bytes):
        raise ValueError("ชนิดไฟล์ไม่ถูกต้อง (รองรับเฉพาะ JPG/PNG)")

    x = load_rgb_224_from_bytes(image_bytes)  # (224,224,3) float32 0..255
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    return x, filename


def _predict_one_model(x: np.ndarray, model_key: str) -> Dict[str, Any]:
    model, meta = get_model_and_meta(model_key)
    y = model.predict(x, verbose=0)

    prob_no = float(np.squeeze(y))
    if np.isnan(prob_no) or np.isinf(prob_no):
        raise RuntimeError(f"โมเดล '{model_key}' ให้ผลลัพธ์ไม่ถูกต้อง")
    prob_no = float(np.clip(prob_no, 0.0, 1.0))
    prob_dr = 1.0 - prob_no

    if prob_dr >= THRESHOLD:
        pred = "DR"
        confidence = prob_dr
    else:
        pred = "No DR"
        confidence = prob_no

    return {
        "model": model_key,
        "model_display_name": meta.display_name,
        "prediction": pred,
        "confidence": _round4(confidence),
        "probabilities": {"No DR": _round4(prob_no), "DR": _round4(prob_dr)},
    }


def predict_image_bytes(image_bytes: bytes, original_name: str, model_key: Optional[str] = None) -> Dict[str, Any]:
    """Single-model prediction (default mode) — pick one model, predict one/many images with it."""
    x, filename = _prepare_input(image_bytes, original_name)
    key = model_key or DEFAULT_MODEL_KEY

    result = _predict_one_model(x, key)
    result["filename"] = filename

    logger.info(
        "ts=%s filename=%s model=%s pred=%s confidence=%.4f prob_dr=%.4f",
        _utc_iso(), filename, key, result["prediction"],
        float(result["confidence"]), float(result["probabilities"]["DR"]),
    )
    return result


def predict_image_bytes_compare(image_bytes: bytes, original_name: str) -> Dict[str, Any]:
    """Compare mode — run every successfully-loaded model against the same image."""
    x, filename = _prepare_input(image_bytes, original_name)

    available = list_available_models()  # key -> display_name, loaded models only
    if not available:
        raise RuntimeError("ไม่มีโมเดลใดพร้อมใช้งาน (ตรวจสอบไฟล์ .keras ใน app/models/)")

    per_model: List[Dict[str, Any]] = []
    for key in available:
        try:
            per_model.append(_predict_one_model(x, key))
        except Exception as e:
            logger.exception("Model '%s' failed during compare mode", key)
            per_model.append({
                "model": key,
                "model_display_name": MODEL_REGISTRY.get(key, {}).get("display_name", key),
                "error": str(e),
            })

    logger.info("ts=%s filename=%s compare_models=%s", _utc_iso(), filename, list(available))

    return {"filename": filename, "models": per_model}


def evaluate_batch(
    images: List[Tuple[bytes, str]],
    labels: Dict[str, int],
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    ประเมินประสิทธิภาพโมเดล (Accuracy/Precision/Recall/Specificity/F1/AUC + ROC curve)
    จากชุดภาพที่มี label จริงกำกับไว้แล้ว — ต่างจาก predict_image_bytes/_compare ที่ทำนาย
    ภาพเดี่ยวๆ โดยไม่รู้คำตอบจริง ฟังก์ชันนี้ใช้สำหรับประเมินผลบนชุดข้อมูลทดสอบ

    images: list ของ (image_bytes, original_filename)
    labels: dict sanitized_filename -> 0|1 (1 = DR) ที่ resolve มาแล้วจาก
            app.utils.eval_utils.resolve_labels
    model_keys: รายชื่อโมเดลที่จะรัน — None แปลว่ารันทุกโมเดลที่โหลดสำเร็จ
    """
    available = list_available_models()  # key -> display_name, loaded models only
    if not available:
        raise RuntimeError("ไม่มีโมเดลใดพร้อมใช้งาน (ตรวจสอบไฟล์ .keras ใน app/models/)")

    keys_to_run = model_keys if model_keys else list(available.keys())
    unknown = [k for k in keys_to_run if k not in available]
    if unknown:
        raise ValueError(f"ไม่รู้จักโมเดลหรือโมเดลยังไม่พร้อมใช้งาน: {', '.join(unknown)}")

    # เตรียมภาพ + จับคู่กับ label ล่วงหน้าครั้งเดียว (ไม่ต้องเปิด/ตรวจสอบไฟล์ซ้ำต่อโมเดล)
    prepared: List[Tuple[np.ndarray, str, int]] = []
    skipped: List[Dict[str, str]] = []

    for image_bytes, original_name in images:
        try:
            x, filename = _prepare_input(image_bytes, original_name)
        except ValueError as e:
            skipped.append({"filename": sanitize_filename(original_name), "reason": str(e)})
            continue

        if filename not in labels:
            skipped.append({"filename": filename, "reason": "ไม่พบ label สำหรับไฟล์นี้"})
            continue

        prepared.append((x, filename, labels[filename]))

    if not prepared:
        raise ValueError("ไม่มีภาพที่จับคู่กับ label ได้เลย ตรวจสอบไฟล์ CSV หรือชื่อไฟล์อีกครั้ง")

    per_model_results: List[Dict[str, Any]] = []
    for key in keys_to_run:
        y_true: List[int] = []
        y_prob_dr: List[float] = []
        per_image: List[Dict[str, Any]] = []
        model_failed = False

        for x, filename, true_label in prepared:
            try:
                pred = _predict_one_model(x, key)
            except Exception as e:
                # เกิด error กับโมเดลนี้ระหว่างทำนาย -> บันทึก error เฉพาะโมเดลนี้
                # แล้วข้ามไปโมเดลถัดไปเลย (เหมือน pattern ใน predict_image_bytes_compare)
                logger.exception("Model '%s' failed during batch evaluation on '%s'", key, filename)
                model_failed = True
                per_model_results.append({
                    "model": key,
                    "model_display_name": MODEL_REGISTRY.get(key, {}).get("display_name", key),
                    "error": str(e),
                })
                break

            prob_dr = float(pred["probabilities"]["DR"])
            y_true.append(true_label)
            y_prob_dr.append(prob_dr)
            per_image.append({
                "filename": filename,
                "true_label": "DR" if true_label == 1 else "No DR",
                "predicted": pred["prediction"],
                "prob_dr": prob_dr,
                "correct": (pred["prediction"] == "DR") == (true_label == 1),
            })

        if model_failed:
            continue

        meta = get_model_and_meta(key)[1]
        metrics = compute_metrics(y_true, y_prob_dr, threshold=THRESHOLD, model_label=meta.display_name)
        per_model_results.append({
            "model": key,
            "model_display_name": meta.display_name,
            "metrics": metrics,
            "per_image": per_image,
        })

    logger.info(
        "ts=%s batch_eval n_images=%d n_skipped=%d models=%s",
        _utc_iso(), len(prepared), len(skipped), keys_to_run,
    )

    return {
        "n_images_used": len(prepared),
        "skipped": skipped,
        "threshold": THRESHOLD,
        "results": per_model_results,
    }


def get_system_info() -> Dict[str, Any]:
    metas = get_all_meta()
    any_ok = False
    models_info = []
    for key, meta in metas.items():
        if meta.status == "ok":
            any_ok = True
        models_info.append({
            "key": key,
            "display_name": meta.display_name,
            "status": meta.status,
            "model_version": meta.model_version,
            "last_updated": meta.last_updated,
            "error": meta.error,
        })

    return {
        "status": "ok" if any_ok else "error",
        "threshold": THRESHOLD,
        "default_model": DEFAULT_MODEL_KEY,
        "models": models_info,
    }
