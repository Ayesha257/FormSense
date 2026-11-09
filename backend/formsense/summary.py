"""Presentation helpers for the UI and reports: checklist, plain-English summary, crops, downloads.

No Streamlit in here, so everything can be unit tested and reused by the report generator.
"""
from __future__ import annotations

import csv
import io
from collections import Counter, OrderedDict
from typing import Sequence

import cv2
import numpy as np

from formsense.schema import (FieldResult, FieldSpec, FieldStatus, FieldType, FormTemplate,
                              Issue, Severity)

SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
SECTION_TITLES = {"personal": "Personal details", "program": "Program applied for", "academic": "Academic record",
                  "declaration": "Declaration"}
STATUS_ICON = {"ok": "✅", "missing": "❌", "unreadable": "❓", "invalid": "⚠️"}


def sort_issues(issues: Sequence[Issue]) -> list[Issue]:
    """Errors first, then warnings, then info; stable inside each group."""
    return sorted(issues, key=lambda i: SEVERITY_ORDER.get(i.severity, 9))


def issue_counts(issues: Sequence[Issue]) -> dict[str, int]:
    c = Counter(i.severity.value for i in issues)
    return {"error": c.get("error", 0), "warning": c.get("warning", 0), "info": c.get("info", 0)}


def field_counts(fields: Sequence[FieldResult], template: FormTemplate) -> dict[str, int]:
    """Items on the form by outcome. A tick-box group counts once (exactly one must be ticked)."""
    rows = checklist(fields, template, {})
    c = Counter(r["status"] for r in rows)
    return {k: c.get(k, 0) for k in ("ok", "missing", "unreadable", "invalid")}


# ---------------- checklist ----------------

def checklist(fields: Sequence[FieldResult], template: FormTemplate,
              labels: dict[str, str]) -> list[dict]:
    """One row per form item, in form order: {section, item, status, detail}.

    Text, date and signature fields give one row each. Tick boxes of one group
    (gender, program) give a single row that is ok only when exactly one is ticked.
    """
    by_name = {f.name: f for f in fields}
    rows: list[dict] = []
    seen_groups: set[str] = set()
    for spec in template.fields:
        r = by_name.get(spec.name)
        if r is None:
            continue
        section = SECTION_TITLES.get(spec.section or "", (spec.section or "Other").title())
        if spec.type == FieldType.CHECKBOX and spec.group:
            if spec.group in seen_groups:
                continue
            seen_groups.add(spec.group)
            members = [(s, by_name[s.name]) for s in template.fields
                       if s.group == spec.group and s.name in by_name]
            ticked = [s.label for s, res in members if res.checked]
            if len(ticked) == 1:
                status, detail = "ok", ticked[0]
            elif not ticked:
                status, detail = "missing", "nothing ticked"
            else:
                status, detail = "invalid", "more than one ticked: " + ", ".join(ticked)
            rows.append({"section": section, "item": labels.get(spec.name, spec.group.title()),
                         "status": status, "detail": detail})
            continue
        rows.append({"section": section, "item": labels.get(spec.name, spec.label),
                     "status": r.status.value, "detail": _detail(spec, r)})
    return rows


def _detail(spec: FieldSpec, r: FieldResult) -> str:
    if spec.type == FieldType.CHECKBOX:
        return "ticked" if r.checked else "not ticked"
    if spec.type == FieldType.SIGNATURE:
        return "signature present" if r.status == FieldStatus.OK else "signature box is empty"
    if r.status == FieldStatus.MISSING:
        return "left empty"
    text = r.raw_text or r.value or ""
    if r.status == FieldStatus.UNREADABLE:
        return f"hard to read{' (looks like: ' + text + ')' if text else ''}"
    return text


def checklist_csv(rows: Sequence[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["section", "item", "status", "detail"])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ---------------- plain-English summary (no LLM) ----------------

def plain_summary(fields: Sequence[FieldResult], issues: Sequence[Issue], template: FormTemplate,
                  labels: dict[str, str]) -> str:
    """A short deterministic verdict. The LLM explanation, when available, replaces it in the UI."""
    counts = issue_counts(issues)
    if not issues:
        return "No problems found: every required item is filled in and the values agree with each other."
    parts = []
    if counts["error"]:
        parts.append(f"{counts['error']} error{'s' if counts['error'] != 1 else ''}")
    if counts["warning"]:
        parts.append(f"{counts['warning']} warning{'s' if counts['warning'] != 1 else ''}")
    if counts["info"]:
        parts.append(f"{counts['info']} note{'s' if counts['info'] != 1 else ''}")
    lines = [f"This form has {' and '.join(parts)} to fix before it can be accepted:"]
    for i in sort_issues(issues):
        who = ", ".join(OrderedDict.fromkeys(labels.get(f, f) for f in i.fields)) or "Form"
        lines.append(f"- **{who}**: {i.message}")
    return "\n".join(lines)


# ---------------- images ----------------

def crop_field(aligned: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 14) -> np.ndarray:
    """A padded crop around a field box (grayscale or BGR), clipped to the image."""
    x, y, w, h = bbox
    H, W = aligned.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return np.full((40, 120), 255, dtype=np.uint8)
    return aligned[y0:y1, x0:x1].copy()


def highlight(img_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
              color=(40, 40, 220)) -> np.ndarray:
    """Translucent highlight over evidence boxes, for 'show me where the problem is'."""
    out = img_bgr.copy()
    overlay = out.copy()
    pad = 6
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(overlay, (x - pad, y - pad), (x + w + pad, y + h + pad), color, -1)
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 3)
    return cv2.addWeighted(overlay, 0.25, out, 0.75, 0)


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return buf.tobytes()
