from __future__ import annotations

import base64
import csv
import io
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # ต้องตั้งก่อน import pyplot เพื่อไม่ให้พยายามเปิดหน้าต่าง GUI บน server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from app.utils.image_utils import sanitize_filename

# ป้ายกำกับที่ถือว่าหมายถึง "เป็น DR" (label = 1) เมื่อ parse จาก CSV หรือ prefix ชื่อไฟล์
_POSITIVE_TOKENS = {"dr", "1", "yes", "true", "positive", "pos"}
_NEGATIVE_TOKENS = {"no dr", "no_dr", "nodr", "0", "no", "false", "negative", "neg", "normal"}

# prefix ที่จะจับได้จากชื่อไฟล์ เรียงให้ "no_dr"/"nodr" ถูกจับก่อน "dr" เดี่ยวๆ เสมอ
_PREFIX_RE = re.compile(r"^(no[_\-]?dr|dr|normal)[_\-]", re.IGNORECASE)


def _normalize_label_token(raw: str) -> Optional[int]:
    token = (raw or "").strip().lower()
    if token in _POSITIVE_TOKENS:
        return 1
    if token in _NEGATIVE_TOKENS:
        return 0
    return None


def parse_labels_csv(csv_bytes: bytes) -> Dict[str, int]:
    """
    อ่านไฟล์ CSV ที่มีคอลัมน์ filename,label (ชื่อคอลัมน์ไม่สนตัวพิมพ์เล็กใหญ่)
    label ยอมรับได้ทั้ง DR/No DR, 1/0, yes/no ฯลฯ
    คืนค่า dict: sanitized_filename -> 0|1  (1 = DR)
    """
    text = csv_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("ไฟล์ CSV ว่างเปล่าหรืออ่านหัวตารางไม่ได้")

    field_map = {f.lower().strip(): f for f in reader.fieldnames}
    filename_col = field_map.get("filename") or field_map.get("file") or field_map.get("image")
    label_col = field_map.get("label") or field_map.get("class") or field_map.get("y")
    if not filename_col or not label_col:
        raise ValueError(
            "ไฟล์ CSV ต้องมีคอลัมน์ 'filename' และ 'label' (รองรับชื่อสำรอง: file/image, class/y)"
        )

    mapping: Dict[str, int] = {}
    for row_num, row in enumerate(reader, start=2):
        fname_raw = (row.get(filename_col) or "").strip()
        label_raw = (row.get(label_col) or "").strip()
        if not fname_raw:
            continue
        label_val = _normalize_label_token(label_raw)
        if label_val is None:
            raise ValueError(
                f"CSV แถวที่ {row_num}: ไม่เข้าใจค่า label '{label_raw}' "
                f"(ใช้ได้: DR/No DR หรือ 1/0)"
            )
        mapping[sanitize_filename(fname_raw)] = label_val

    if not mapping:
        raise ValueError("ไม่พบข้อมูล label ในไฟล์ CSV")
    return mapping


def label_from_filename_prefix(filename: str) -> Optional[int]:
    """
    เดา label จาก prefix ของชื่อไฟล์ เช่น DR_0001.jpg -> 1, NoDR_0002.jpg / No_DR_0002.jpg -> 0
    คืน None ถ้าเดาไม่ได้ (ให้ caller ตัดสินใจว่าจะข้ามหรือ error)
    """
    m = _PREFIX_RE.match(filename)
    if not m:
        return None
    token = m.group(1).lower().replace("-", "_")
    if token == "dr":
        return 1
    if token in {"no_dr", "nodr", "normal"}:
        return 0
    return None


def resolve_labels(
    filenames: List[str],
    csv_bytes: Optional[bytes],
) -> Tuple[Dict[str, int], List[str]]:
    """
    รวมสองรูปแบบ label input เข้าด้วยกัน:
      - ถ้ามี CSV แนบมา ใช้ CSV เป็นหลัก
      - ถ้าไม่มี CSV (หรือไฟล์ไหนหาไม่เจอใน CSV) ลองเดาจาก prefix ชื่อไฟล์แทน
    คืนค่า (mapping sanitized_filename -> 0|1, รายชื่อไฟล์ต้นฉบับที่ resolve label ไม่ได้เลย)
    """
    csv_map: Dict[str, int] = {}
    if csv_bytes:
        csv_map = parse_labels_csv(csv_bytes)

    resolved: Dict[str, int] = {}
    missing: List[str] = []

    for original in filenames:
        clean = sanitize_filename(original)
        if clean in csv_map:
            resolved[clean] = csv_map[clean]
            continue
        guessed = label_from_filename_prefix(clean)
        if guessed is not None:
            resolved[clean] = guessed
            continue
        missing.append(original)

    return resolved, missing


def _round4(x: float) -> float:
    return float(f"{float(x):.4f}")


def _render_roc_png(fpr: np.ndarray, tpr: np.ndarray, auc_value: float, label: str = "") -> str:
    """วาด ROC curve ด้วย matplotlib แล้วคืนเป็น base64 PNG string สำหรับฝัง <img src='data:image/png;base64,...'>"""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    title_suffix = f" — {label}" if label else ""
    ax.plot(fpr, tpr, color="#5eead4", linewidth=2, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=1, linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve{title_suffix}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_metrics(
    y_true: List[int],
    y_prob_dr: List[float],
    threshold: float = 0.5,
    model_label: str = "",
) -> Dict[str, object]:
    """
    คำนวณ Accuracy, Precision, Recall(=Sensitivity), Specificity, F1, AUC, Confusion Matrix
    แล้ววาด ROC curve เป็น base64 PNG (ฝังไว้ใน result["roc_curve_png_base64"])

    y_true: 0|1 จริง (1 = DR)
    y_prob_dr: ความน่าจะเป็นที่โมเดลให้ว่าเป็น DR (ค่าต่อเนื่อง 0..1)
    threshold: ค่าขีดแบ่งสำหรับตัดสินคลาสจาก y_prob_dr (ค่าเริ่มต้น 0.5)
    model_label: ชื่อโมเดล ใส่เป็นหัวกราฟ ROC (ไม่บังคับ)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob_dr, dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    n_pos = int(y_true_arr.sum())
    n_neg = int(len(y_true_arr) - n_pos)

    result: Dict[str, object] = {
        "n_samples": int(len(y_true_arr)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": threshold,
        "accuracy": _round4(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": _round4(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": _round4(recall_score(y_true_arr, y_pred_arr, zero_division=0)),  # = Sensitivity
        "f1": _round4(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }
    # Sensitivity เป็นชื่อเดียวกับ Recall ในบริบทนี้ (positive class = DR) — แสดงซ้ำให้ตรงศัพท์ทางการแพทย์
    result["sensitivity"] = result["recall"]

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    result["confusion_matrix"] = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    # Specificity = TN / (TN + FP) — คู่ตรงข้ามของ Recall/Sensitivity มีประโยชน์ทางคลินิก
    result["specificity"] = _round4(tn / (tn + fp)) if (tn + fp) > 0 else None

    # AUC/ROC ต้องมีทั้งสองคลาสอยู่ในชุดข้อมูล ไม่งั้นคำนวณไม่ได้ (sklearn จะ error)
    if n_pos > 0 and n_neg > 0:
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        result["auc"] = _round4(auc(fpr, tpr))
        result["roc_curve_png_base64"] = _render_roc_png(fpr, tpr, result["auc"], label=model_label)
    else:
        result["auc"] = None
        result["roc_curve_png_base64"] = None
        result["auc_note"] = (
            "คำนวณ AUC/ROC ไม่ได้ เนื่องจากชุดข้อมูลมีคลาสเดียว "
            "(ต้องมีทั้งภาพ DR และ No DR อย่างน้อยอย่างละ 1 ภาพ)"
        )

    return result
