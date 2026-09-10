from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf

from app.models.registry import MODEL_REGISTRY

logger = logging.getLogger("app.model_loader")


@dataclass(frozen=True)
class ModelMeta:
    key: str
    display_name: str
    model_path: str
    model_version: Optional[str]
    last_updated: Optional[str]
    status: str = "ok"  # "ok" | "error"
    error: Optional[str] = None


_LOCK = threading.Lock()
_MODELS: Dict[str, "tf.keras.Model"] = {}
_META: Dict[str, ModelMeta] = {}

# First key declared in the registry acts as the default when the caller
# (e.g. /predict without a `model` field) doesn't specify one.
DEFAULT_MODEL_KEY = next(iter(MODEL_REGISTRY))


def _models_dir() -> Path:
    """
    Folder that holds every .keras file. Override with MODELS_DIR env var
    if you want to keep the weights outside the repo. Defaults to
    app/models/ (next to this file).
    """
    env = os.getenv("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent / "models").resolve()


def _iso_utc_from_mtime(mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _sha256_short(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _load_one(key: str) -> ModelMeta:
    """Load a single model by registry key. Caller must hold _LOCK."""
    cfg = MODEL_REGISTRY[key]
    model_path = _models_dir() / cfg["file"]

    if not model_path.is_file():
        meta = ModelMeta(
            key=key,
            display_name=cfg["display_name"],
            model_path=str(model_path),
            model_version=None,
            last_updated=None,
            status="error",
            error=f"ไม่พบไฟล์โมเดล: {model_path}",
        )
        logger.error("Model file missing for '%s': %s", key, model_path)
        _META[key] = meta
        return meta

    try:
        logger.info("Loading model '%s' from %s", key, model_path)
        model = tf.keras.models.load_model(str(model_path), compile=False)
        stat = model_path.stat()
        meta = ModelMeta(
            key=key,
            display_name=cfg["display_name"],
            model_path=str(model_path),
            model_version=_sha256_short(model_path),
            last_updated=_iso_utc_from_mtime(stat.st_mtime),
            status="ok",
        )
        _MODELS[key] = model
        _META[key] = meta
        logger.info(
            "Model '%s' loaded OK (version=%s updated=%s)",
            key, meta.model_version, meta.last_updated,
        )
        return meta
    except Exception as e:
        logger.exception("Failed to load model '%s'", key)
        meta = ModelMeta(
            key=key,
            display_name=cfg["display_name"],
            model_path=str(model_path),
            model_version=None,
            last_updated=None,
            status="error",
            error=str(e),
        )
        _META[key] = meta
        return meta


def warmup_all_models() -> None:
    """
    Load every model declared in the registry once, at server startup.
    A model failing to load never stops the others — it just gets marked
    status="error" and is skipped everywhere else (dropdown, compare mode).
    """
    with _LOCK:
        for key in MODEL_REGISTRY:
            if key not in _MODELS and key not in _META:
                _load_one(key)


def get_model_and_meta(model_key: Optional[str] = None) -> Tuple["tf.keras.Model", ModelMeta]:
    """Return (model, meta) for one model, loading it on demand if needed."""
    key = model_key or DEFAULT_MODEL_KEY
    if key not in MODEL_REGISTRY:
        raise ValueError(f"ไม่รู้จักโมเดล '{key}'")

    if key in _MODELS and _META.get(key) and _META[key].status == "ok":
        return _MODELS[key], _META[key]

    with _LOCK:
        if key in _MODELS and _META.get(key) and _META[key].status == "ok":
            return _MODELS[key], _META[key]
        meta = _load_one(key)
        if meta.status != "ok":
            raise RuntimeError(meta.error or f"โหลดโมเดล '{key}' ไม่สำเร็จ")
        return _MODELS[key], meta


def get_all_meta() -> Dict[str, ModelMeta]:
    """Metadata for every registered model (loaded or failed). Used by /health, /admin, /models."""
    with _LOCK:
        for key in MODEL_REGISTRY:
            if key not in _META:
                _load_one(key)
        return dict(_META)


def list_available_models() -> Dict[str, str]:
    """key -> display_name, for models that loaded successfully only."""
    metas = get_all_meta()
    return {k: m.display_name for k, m in metas.items() if m.status == "ok"}
