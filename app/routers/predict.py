from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.inference import (
    THRESHOLD,
    get_system_info,
    predict_image_bytes,
    predict_image_bytes_compare,
)
from app.model_loader import DEFAULT_MODEL_KEY, list_available_models

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
