from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

logger = logging.getLogger("app.model_loader")


@dataclass(frozen=True)
class ModelMeta:
    model_path: str
    model_version: str
    last_updated: str


_LOCK = threading.Lock()
_MODEL: Optional[tf.keras.Model] = None
_META: Optional[ModelMeta] = None


def _iso_utc_from_mtime(mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _sha256_short(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _resolve_model_path() -> Path:
    env = os.getenv("MODEL_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))

    base_dir = Path(__file__).resolve().parent
    candidates.append(base_dir / "best_model.keras")
    candidates.append(Path.cwd() / "best_model.keras")

    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        if rp.is_file():
            return rp

    raise FileNotFoundError(
        "ไม่พบไฟล์โมเดล best_model.keras กรุณาวางไฟล์ไว้ที่ `app/best_model.keras` "
        "หรือกำหนด environment variable `MODEL_PATH` ให้ชี้ไปยังไฟล์โมเดล"
    )


def get_model_and_meta() -> Tuple[tf.keras.Model, ModelMeta]:
    global _MODEL, _META

    if _MODEL is not None and _META is not None:
        return _MODEL, _META

    with _LOCK:
        if _MODEL is not None and _META is not None:
            return _MODEL, _META

        model_path = _resolve_model_path()
        stat = model_path.stat()

        meta = ModelMeta(
            model_path=str(model_path),
            model_version=_sha256_short(model_path),
            last_updated=_iso_utc_from_mtime(stat.st_mtime),
        )

        logger.info("Loading model from %s", meta.model_path)
        model = tf.keras.models.load_model(str(model_path), compile=False)
        _MODEL = model
        _META = meta
        logger.info("Model loaded OK (version=%s updated=%s)", meta.model_version, meta.last_updated)
        return model, meta


def warmup_model() -> None:
    try:
        get_model_and_meta()
    except Exception:
        logger.exception("Model warmup failed; server will start but /predict will fail until fixed.")

