from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.inference import (
    THRESHOLD,
    evaluate_batch,
    get_system_info,
    predict_image_bytes,
    predict_image_bytes_compare,
)
from app.model_loader import DEFAULT_MODEL_KEY, list_available_models
from app.utils.eval_utils import resolve_labels
from app.utils.image_utils import sanitize_filename

logger = logging.getLogger("app.routers.predict")

router = APIRouter()

_BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


def _models_for_frontend() -> List[dict]:
    available = list_available_models()
    return [{"key": k, "display_name": v} for k, v in available.items()]


@router.get("/health")
def health():
    return get_system_info()


@router.get("/models")
def models():
    """Loaded models the frontend can offer (dropdown / compare mode)."""
    return {
        "default_model": DEFAULT_MODEL_KEY,
        "models": _models_for_frontend(),
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    sys = get_system_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": sys.get("status"),
            "threshold": sys.get("threshold", THRESHOLD),
            "default_model": DEFAULT_MODEL_KEY,
            "models": _models_for_frontend(),
        },
    )


@router.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    sys = get_system_info()
    return templates.TemplateResponse("admin.html", {"request": request, "system": sys})


@router.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    model: Optional[str] = Form(default=None),
):
    """Single-model mode: predict one or more images using ONE chosen model."""
    if not files:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์อัปโหลด")

    results = []
    try:
        for f in files:
            content = await f.read()
            results.append(predict_image_bytes(content, f.filename or "image", model_key=model))
    except ValueError as e:
        logger.warning("Validation error filename=%s msg=%s", getattr(f, "filename", None), str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Internal error during prediction")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดภายในระบบ") from e

    if len(results) == 1:
        return results[0]
    return {"results": results, "count": len(results), "threshold": THRESHOLD}


@router.post("/evaluate")
async def evaluate(
    files_dr: Optional[List[UploadFile]] = File(default=None),
    files_nodr: Optional[List[UploadFile]] = File(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    labels_csv: Optional[UploadFile] = File(default=None),
    model: Optional[str] = Form(default=None),
    compare_all: bool = Form(default=False),
):
    """
    ประเมินประสิทธิภาพโมเดล (Accuracy/Precision/Recall/Specificity/F1/AUC + ROC curve)
    จากชุดภาพที่รู้คำตอบจริง ต่างจาก /predict และ /predict/compare ที่ทำนายโดยไม่รู้คำตอบจริง

    แหล่ง label (เรียงตามความสำคัญ):
      1) files_dr / files_nodr : label มาจาก "ช่องที่วางภาพ" ไม่ต้องดูชื่อไฟล์ (แนะนำ)
      2) labels_csv            : ใช้กับภาพใน `files` (CSV คอลัมน์ filename,label)
      3) prefix ชื่อไฟล์        : ใช้กับภาพใน `files` ที่ไม่อยู่ใน CSV (DR_xxx / NoDR_xxx)

    - model: โมเดลเดียวที่จะประเมิน (มีผลเฉพาะตอน compare_all=false)
    - compare_all: true = ประเมินทุกโมเดลที่โหลดสำเร็จพร้อมกัน
    """
    files_dr = files_dr or []
    files_nodr = files_nodr or []
    files = files or []

    if not (files_dr or files_nodr or files):
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์ภาพสำหรับประเมินผล")

    images: List[tuple] = []
    labels_map: dict = {}
    missing: List[str] = []
    duplicates: List[str] = []

    def _add_labeled(name: str, content: bytes, label: int) -> None:
        clean = sanitize_filename(name)
        if clean in labels_map:
            # ชื่อซ้ำ (เช่น 2 ช่องมีไฟล์ชื่อเดียวกัน) จะจับคู่ label ผิดได้ จึงข้ามและแจ้งกลับ
            duplicates.append(name)
            return
        labels_map[clean] = label
        images.append((content, name))

    # (1) label จากช่องที่วางภาพ
    for f in files_dr:
        _add_labeled(f.filename or "image", await f.read(), 1)
    for f in files_nodr:
        _add_labeled(f.filename or "image", await f.read(), 0)

    # (2)+(3) ภาพในช่อง files แบบเดิม: CSV ก่อน แล้วค่อยเดาจาก prefix
    if files:
        csv_bytes: Optional[bytes] = None
        if labels_csv is not None:
            csv_bytes = await labels_csv.read()
        try:
            extra_map, missing = resolve_labels([f.filename or "image" for f in files], csv_bytes)
        except ValueError as e:
            logger.warning("Label parsing error: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e

        for f in files:
            content = await f.read()
            clean = sanitize_filename(f.filename or "image")
            if clean not in extra_map:
                continue
            if clean in labels_map:
                duplicates.append(f.filename or "image")
                continue
            labels_map[clean] = extra_map[clean]
            images.append((content, f.filename or "image"))

    if not labels_map:
        raise HTTPException(
            status_code=400,
            detail=(
                "ไม่สามารถระบุ label ของภาพใดได้เลย กรุณาวางภาพในช่อง DR / No DR "
                "หรือแนบ CSV (filename,label) หรือตั้งชื่อไฟล์ให้ขึ้นต้นด้วย DR_ / NoDR_"
            ),
        )

    model_keys: Optional[List[str]] = None
    if not compare_all:
        model_keys = [model or DEFAULT_MODEL_KEY]

    try:
        result = evaluate_batch(images, labels_map, model_keys=model_keys)
    except ValueError as e:
        logger.warning("Validation error during evaluate: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.warning("No models available for evaluate: %s", str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Internal error during batch evaluation")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดภายในระบบระหว่างประเมินผล") from e

    if missing:
        result["unlabeled_files_ignored"] = missing
    if duplicates:
        result["duplicate_filenames_ignored"] = duplicates

    return result


@router.post("/predict/compare")
async def predict_compare(files: List[UploadFile] = File(...)):
    """Compare mode: predict each image with EVERY loaded model, side by side."""
    if not files:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์อัปโหลด")

    results = []
    try:
        for f in files:
            content = await f.read()
            results.append(predict_image_bytes_compare(content, f.filename or "image"))
    except ValueError as e:
        logger.warning("Validation error filename=%s msg=%s", getattr(f, "filename", None), str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Internal error during compare prediction")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดภายในระบบ") from e

    return {"results": results, "count": len(results), "threshold": THRESHOLD}
