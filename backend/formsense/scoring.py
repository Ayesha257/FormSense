"""Score predicted fields and issues against ground truth (shared by the app and scripts).

Ground truth per field: {"name", "status": "ok"|"missing"|"unreadable"|"invalid", "text"?, "checked"?}
Use `truth_from_label` to convert a generator label file (data/synthetic/*/labels/*.json).
"""
from __future__ import annotations

import re
from collections import Counter

from formsense.schema import FieldResult, FieldStatus, FieldType, Issue


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def truth_from_label(label: dict) -> dict[str, dict]:
    """Generator label file -> {field name: truth row}."""
    out = {}
    for f in label["fields"]:
        row = {"name": f["name"], "status": f["status"]}
        if f.get("checked") is not None:
            row["checked"] = bool(f["checked"])
        else:
            row["text"] = f.get("raw_text", "")
        out[f["name"]] = row
    return out


def score_field(kind: FieldType, pred: FieldResult, truth: dict) -> dict:
    """Per-field outcome used by every metric below."""
    row = {"kind": kind.value, "truth_missing": truth["status"] == "missing",
           "truth_unreadable": truth["status"] == "unreadable"}
    if kind == FieldType.CHECKBOX:
        row["checkbox_ok"] = bool(pred.checked) == truth["checked"]
        row["correct"] = row["checkbox_ok"]
        return row
    row["pred_missing"] = pred.status == FieldStatus.MISSING
    if kind == FieldType.SIGNATURE:
        row["correct"] = row["pred_missing"] == row["truth_missing"]
        return row
    row["pred_text"] = pred.raw_text
    row["truth_text"] = truth.get("text", "")
    if row["truth_missing"]:
        row["correct"] = row["pred_missing"]
    elif row["truth_unreadable"]:
        # the ink is deliberately illegible: the right answer is to flag it, not to read it
        row["correct"] = pred.status == FieldStatus.UNREADABLE
    else:
        row["text_ok"] = norm(pred.raw_text) == norm(truth.get("text", ""))
        row["correct"] = row["text_ok"] and not row["pred_missing"]
    return row


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def f1(p: float | None, r: float | None) -> float:
    return round(2 * p * r / (p + r), 1) if p and r else 0.0


def summarise(rows: list[dict]) -> dict:
    text = [r for r in rows if r["kind"] in ("text", "date")
            and not r["truth_missing"] and not r["truth_unreadable"]]
    miss = [r for r in rows if r["kind"] in ("text", "date", "signature")]
    tp = sum(r["pred_missing"] and r["truth_missing"] for r in miss)
    fp = sum(r["pred_missing"] and not r["truth_missing"] for r in miss)
    fn = sum(not r["pred_missing"] and r["truth_missing"] for r in miss)
    prec, rec = pct(tp, tp + fp), pct(tp, tp + fn)
    boxes = [r for r in rows if r["kind"] == "checkbox"]
    sig = [r for r in rows if r["kind"] == "signature"]
    unread = [r for r in rows if r["truth_unreadable"]]
    out = {
        "Text values read exactly (%)": pct(sum(r["text_ok"] for r in text), len(text)),
        "Missing-field precision (%)": prec,
        "Missing-field recall (%)": rec,
        "Missing-field F1 (%)": f1(prec, rec),
        "Checkbox accuracy (%)": pct(sum(r["checkbox_ok"] for r in boxes), len(boxes)),
        "Signature accuracy (%)": pct(sum(r["correct"] for r in sig), len(sig)),
    }
    if unread:
        out["Unreadable fields flagged (%)"] = pct(sum(r["correct"] for r in unread), len(unread))
    out["All fields correct (%)"] = pct(sum(r["correct"] for r in rows), len(rows))
    return out


# ---------------- issues (end to end: vision + rules) ----------------

def issue_key(i: Issue | dict) -> tuple[str, frozenset]:
    d = i if isinstance(i, dict) else i.model_dump(mode="json")
    return d["id"], frozenset(d["fields"])


def score_issues(pred: list[Issue], truth: list[dict]) -> dict:
    """An issue counts as found when the same rule fires on the same fields."""
    p, t = Counter(issue_key(i) for i in pred), Counter(issue_key(i) for i in truth)
    tp = sum((p & t).values())
    return {"tp": tp, "fp": sum(p.values()) - tp, "fn": sum(t.values()) - tp,
            "missed": sorted(f"{k[0]}:{','.join(sorted(k[1]))}" for k in (t - p).elements()),
            "extra": sorted(f"{k[0]}:{','.join(sorted(k[1]))}" for k in (p - t).elements())}


def summarise_issues(per_form: list[dict]) -> dict:
    tp = sum(x["tp"] for x in per_form)
    fp = sum(x["fp"] for x in per_form)
    fn = sum(x["fn"] for x in per_form)
    prec, rec = pct(tp, tp + fp), pct(tp, tp + fn)
    clean = [x for x in per_form if x.get("truth_n", 1) == 0]
    out = {"Issue precision (%)": prec, "Issue recall (%)": rec, "Issue F1 (%)": f1(prec, rec)}
    if clean:
        out["Error-free forms passed with no issue (%)"] = pct(sum(x["fp"] == 0 for x in clean), len(clean))
    return out
