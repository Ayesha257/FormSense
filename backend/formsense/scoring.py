"""Score predicted fields against ground truth (shared by the app and scripts).

Ground truth per field: {"name", "status": "ok"|"missing", "text"?, "checked"?}
"""
from __future__ import annotations

import re

from formsense.schema import FieldResult, FieldStatus, FieldType


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def score_field(kind: FieldType, pred: FieldResult, truth: dict) -> dict:
    """Per-field outcome used by every metric below."""
    row = {"kind": kind.value, "truth_missing": truth["status"] == "missing"}
    if kind == FieldType.CHECKBOX:
        row["checkbox_ok"] = bool(pred.checked) == truth["checked"]
        row["correct"] = row["checkbox_ok"]
        return row
    row["pred_missing"] = pred.status == FieldStatus.MISSING
    if kind == FieldType.SIGNATURE:
        row["correct"] = row["pred_missing"] == row["truth_missing"]
        return row
    if row["truth_missing"]:
        row["correct"] = row["pred_missing"]
    else:
        row["text_ok"] = norm(pred.raw_text) == norm(truth.get("text", ""))
        row["correct"] = row["text_ok"] and not row["pred_missing"]
    row["pred_text"] = pred.raw_text
    row["truth_text"] = truth.get("text", "")
    return row


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def summarise(rows: list[dict]) -> dict:
    text = [r for r in rows if r["kind"] in ("text", "date") and not r["truth_missing"]]
    miss = [r for r in rows if r["kind"] in ("text", "date", "signature")]
    tp = sum(r["pred_missing"] and r["truth_missing"] for r in miss)
    fp = sum(r["pred_missing"] and not r["truth_missing"] for r in miss)
    fn = sum(not r["pred_missing"] and r["truth_missing"] for r in miss)
    prec, rec = pct(tp, tp + fp), pct(tp, tp + fn)
    f1 = round(2 * prec * rec / (prec + rec), 1) if prec and rec else 0.0
    boxes = [r for r in rows if r["kind"] == "checkbox"]
    sig = [r for r in rows if r["kind"] == "signature"]
    return {
        "Text values read exactly (%)": pct(sum(r["text_ok"] for r in text), len(text)),
        "Missing-field precision (%)": prec,
        "Missing-field recall (%)": rec,
        "Missing-field F1 (%)": f1,
        "Checkbox accuracy (%)": pct(sum(r["checkbox_ok"] for r in boxes), len(boxes)),
        "Signature accuracy (%)": pct(sum(r["correct"] for r in sig), len(sig)),
        "All fields correct (%)": pct(sum(r["correct"] for r in rows), len(rows)),
    }
