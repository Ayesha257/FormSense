"""Metrics for comparing a system's output against ground truth, per the brief:

* Field extraction: exact match accuracy, character error rate (CER)
* Missing field detection: precision, recall, F1
* Inconsistency detection: precision, recall, F1 (treated generically as "issue
  detection" so it also covers format and selection issues, since they are scored the
  same way)
* Missing vs unreadable: confusion matrix

Every function here takes plain, already-loaded data (lists of FieldResult / Issue, or
dicts), not file paths, so it is reusable from the harness, from a notebook, and from
tests. File loading lives in harness.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

from formsense.rules.engine import issue_key
from formsense.schema import FieldResult, FieldStatus, Issue


# ============================================================ field extraction
def char_error_rate(pred: str, truth: str) -> float:
    """Levenshtein distance divided by the length of the truth string.

    0.0 is a perfect match. An empty truth string with a non empty prediction scores
    1.0 (capped), not a divide by zero.
    """
    if truth == pred:
        return 0.0
    if not truth:
        return 1.0 if pred else 0.0
    a, b = truth, pred
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return min(prev[len(b)] / len(a), 1.0)


@dataclass
class FieldExtractionScore:
    n: int = 0
    exact_matches: int = 0
    total_cer: float = 0.0
    per_field: Dict[str, "FieldExtractionScore"] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.exact_matches / self.n if self.n else float("nan")

    @property
    def mean_cer(self) -> float:
        return self.total_cer / self.n if self.n else float("nan")

    def as_dict(self) -> dict:
        return {"n": self.n, "exact_match_accuracy": self.accuracy, "mean_cer": self.mean_cer,
                "per_field": {k: {"n": v.n, "exact_match_accuracy": v.accuracy, "mean_cer": v.mean_cer}
                             for k, v in self.per_field.items()}}


def _field_text(f: FieldResult) -> str:
    return (f.value if f.value is not None else f.raw_text) or ""


def score_field_extraction(
    predicted: Sequence[FieldResult], truth: Sequence[FieldResult],
    fields: Sequence[str] = None, skip_statuses: Sequence[FieldStatus] = (FieldStatus.MISSING,),
) -> FieldExtractionScore:
    """Exact match accuracy and CER over field VALUES.

    Only fields whose ground truth status is OK or INVALID are scored by default (a
    MISSING ground truth field has no text to compare). Pass ``fields`` to score only a
    subset, e.g. just the date fields.
    """
    t_by_name = {f.name: f for f in truth}
    p_by_name = {f.name: f for f in predicted}
    names = fields if fields is not None else list(t_by_name)
    out = FieldExtractionScore()
    for name in names:
        tf = t_by_name.get(name)
        if tf is None or tf.status in skip_statuses:
            continue
        pf = p_by_name.get(name)
        t_text, p_text = _field_text(tf), _field_text(pf) if pf else ""
        cer = char_error_rate(p_text, t_text)
        exact = int(p_text == t_text)
        out.n += 1
        out.exact_matches += exact
        out.total_cer += cer
        per = out.per_field.setdefault(name, FieldExtractionScore())
        per.n += 1
        per.exact_matches += exact
        per.total_cer += cer
    return out


# ============================================================ precision / recall / F1
@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:  # NaN-safe: p != p is True only when p is NaN
            return float("nan")
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": self.precision, "recall": self.recall, "f1": self.f1}

    def __add__(self, other: "PRF") -> "PRF":
        return PRF(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def score_missing_field_detection(predicted: Sequence[FieldResult], truth: Sequence[FieldResult]) -> PRF:
    """Treats each field as a binary "is this field MISSING" prediction."""
    t_by_name = {f.name: f for f in truth}
    p_by_name = {f.name: f for f in predicted}
    prf = PRF()
    for name, tf in t_by_name.items():
        t_missing = tf.status == FieldStatus.MISSING
        pf = p_by_name.get(name)
        p_missing = pf is not None and pf.status == FieldStatus.MISSING
        if p_missing and t_missing:
            prf.tp += 1
        elif p_missing and not t_missing:
            prf.fp += 1
        elif not p_missing and t_missing:
            prf.fn += 1
    return prf


def score_issue_detection(predicted: Sequence[Issue], truth: Sequence[Issue],
                          types: Sequence[str] = None) -> PRF:
    """Precision/recall/F1 over issues, matched by (rule id, exact field set).

    ``types`` restricts to IssueType values (e.g. ["inconsistency"]); None scores all
    issue types together.
    """
    keep = lambda issues: {issue_key(i) for i in issues if types is None or i.type.value in types}
    want, got = keep(truth), keep(predicted)
    return PRF(tp=len(want & got), fp=len(got - want), fn=len(want - got))


def aggregate_prf(scores: Sequence[PRF]) -> PRF:
    out = PRF()
    for s in scores:
        out = out + s
    return out


# ============================================================ missing vs unreadable
@dataclass
class ConfusionMatrix:
    labels: Tuple[str, ...]
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def get(self, truth_label: str, pred_label: str) -> int:
        return self.counts.get((truth_label, pred_label), 0)

    def as_rows(self) -> List[List[int]]:
        return [[self.get(t, p) for p in self.labels] for t in self.labels]

    def as_dict(self) -> dict:
        return {"labels": list(self.labels), "rows": self.as_rows()}


def score_missing_vs_unreadable(predicted: Sequence[FieldResult], truth: Sequence[FieldResult]) -> ConfusionMatrix:
    """Confusion matrix restricted to fields the ground truth marked MISSING or UNREADABLE.

    A field the ground truth marks OK or INVALID is out of scope for this particular
    metric (the brief asks specifically about missing-versus-unreadable confusion).
    """
    labels = ("missing", "unreadable")
    t_by_name = {f.name: f for f in truth}
    p_by_name = {f.name: f for f in predicted}
    cm = ConfusionMatrix(labels=labels)
    for name, tf in t_by_name.items():
        if tf.status not in (FieldStatus.MISSING, FieldStatus.UNREADABLE):
            continue
        pf = p_by_name.get(name)
        p_label = pf.status.value if pf and pf.status.value in labels else "other"
        t_label = tf.status.value
        key = (t_label, p_label)
        cm.counts[key] = cm.counts.get(key, 0) + 1
    if any(p == "other" for _, p in cm.counts):
        cm = ConfusionMatrix(labels=labels + ("other",), counts=cm.counts)
    return cm
