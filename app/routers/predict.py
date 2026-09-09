from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.inference import THRESHOLD, get_system_info, predict_image_bytes

logger = logging.getLogger("app.routers.predict")

router = APIRouter()

_BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


@router.get("/health")
def health():
    return get_system_info()


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    sys = get_system_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_version": sys.get("model_version"),
            "last_updated": sys.get("last_updated"),
            "threshold": sys.get("threshold", THRESHOLD),
            "status": sys.get("status"),
        },
    )


@router.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    sys = get_system_info()
    return templates.TemplateResponse("admin.html", {"request": request, "system": sys})


@router.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์อัปโหลด")

    results = []
    try:
        for f in files:
            content = await f.read()
            results.append(predict_image_bytes(content, f.filename or "image"))
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

