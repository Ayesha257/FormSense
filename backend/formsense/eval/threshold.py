"""Choosing the OCR-confidence cutoff for "unreadable" from data instead of by feel.

A text box with ink is UNREADABLE when its OCR confidence is below a cutoff
(formsense/thresholds.py, OCR_UNREADABLE_BELOW; applied by the extraction step inside audit()).
Too high: good reads are flagged and every one raises an `unreadable_field` issue (false alarms, low
precision). Too low: real scribble is read as text (missed unreadable). The cutoff only decides how
saved readings are LABELLED, so it can be swept after the fact on stored readings, without running
OCR again: each cutoff just re-runs audit() with ExtractionConfig(ocr_conf_min=cutoff).

Tune on the dev split, pick the best cutoff, then report once on the test split.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from formsense.eval.harness import LabelledForm, aggregate_results
from formsense.nlp.extract import ExtractionConfig
from formsense.rules.audit import audit
from formsense.schema import FieldResult, FieldStatus

DEFAULT_THRESHOLDS = tuple(round(0.3 + 0.05 * i, 2) for i in range(13))   # 0.30 .. 0.90


def sweep(items: Iterable[Tuple[LabelledForm, Sequence[FieldResult]]], template: Path | str,
          thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> list[dict]:
    """One row per cutoff: field accuracy, issue P/R/F1, and how the 'unreadable' issues fare."""
    items = list(items)
    rows = []
    for t in thresholds:
        cfg = ExtractionConfig(ocr_conf_min=t)
        pairs = [(form, audit(raw, template=template, today=form.reference_date or dt.date.today(), cfg=cfg))
                 for form, raw in items]
        r = aggregate_results(pairs)
        unr = r.issue_detection_by_type["unreadable"]
        flagged = sum(1 for _, o in pairs for f in o.fields if f.status == FieldStatus.UNREADABLE)
        rows.append({"cutoff": t, "field_accuracy": r.field_extraction.accuracy,
                     "issue_precision": r.issue_detection.precision, "issue_recall": r.issue_detection.recall,
                     "issue_f1": r.issue_detection.f1, "unreadable_precision": unr.precision,
                     "unreadable_recall": unr.recall, "unreadable_f1": unr.f1,
                     "unreadable_false_alarms": unr.fp, "fields_flagged_unreadable": flagged})
    return rows


def best(rows: Sequence[dict], key: str = "issue_f1") -> dict:
    """Highest ``key``; ties go to the LOWER cutoff (fewer good reads thrown away)."""
    ok = [r for r in rows if r[key] == r[key]]          # drop NaN
    return max(ok, key=lambda r: (round(r[key], 6), -r["cutoff"])) if ok else rows[0]


def format_markdown(rows: Sequence[dict]) -> str:
    n = lambda x: "n/a" if x != x else f"{x:.2f}"        # noqa: E731
    b = best(rows)
    L = ["### OCR confidence cutoff for \"unreadable\"", "",
         "| Cutoff | Field accuracy | Issue P | Issue R | Issue F1 | Unreadable P | Unreadable R | False alarms | Flagged unreadable |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        mark = " **<- best**" if r is b else ""
        L.append(f"| {r['cutoff']:.2f}{mark} | {n(r['field_accuracy'])} | {n(r['issue_precision'])} | "
                 f"{n(r['issue_recall'])} | {n(r['issue_f1'])} | {n(r['unreadable_precision'])} | "
                 f"{n(r['unreadable_recall'])} | {r['unreadable_false_alarms']} | {r['fields_flagged_unreadable']} |")
    L += ["", f"Best issue F1 at cutoff {b['cutoff']:.2f}. Choose it on the dev split, set "
              "`OCR_UNREADABLE_BELOW` in `formsense/thresholds.py`, then report on the test split.", ""]
    return "\n".join(L)
