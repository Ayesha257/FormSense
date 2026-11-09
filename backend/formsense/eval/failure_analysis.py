"""Where does the accuracy go? Turns (ground truth, system output) pairs into an error breakdown.

A low headline score says THAT the system is wrong; this says WHERE and WHY:

* every field gets one outcome: ok, misread (OCR text differs), lost_missing (content, but the
  system saw an empty box), lost_unreadable (content, flagged unreadable), false_content (stray ink
  read as text), unreadable_not_flagged, missed/false tick, missed/false signature
* every wrong issue (missed or false alarm) is traced back: did one of the fields it is about get
  read wrongly? If so the rule did its job on bad input ("upstream"), if not the rule logic itself
  is at fault. That split tells the team whether to fix OCR/vision or the rules.

Pure functions over FieldResult / Issue lists: no files, no models, easy to unit test.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from formsense.eval.harness import LabelledForm, SystemOutput
from formsense.eval.metrics import _field_text, char_error_rate
from formsense.rules.engine import issue_key
from formsense.schema import FieldResult, FieldStatus, FieldType

OK = "ok"
OUTCOMES = ("ok", "misread", "lost_missing", "lost_unreadable", "false_content", "unreadable_not_flagged",
            "missed_tick", "false_tick", "missed_signature", "false_signature")
OUTCOME_HELP = {
    "ok": "correct",
    "misread": "OCR read different text",
    "lost_missing": "content present, system saw an empty box",
    "lost_unreadable": "content present, system flagged it unreadable",
    "false_content": "box is empty, system read text (stray ink)",
    "unreadable_not_flagged": "truly unreadable, system did not flag it",
    "missed_tick": "box ticked, system saw it empty",
    "false_tick": "box empty, system saw a tick",
    "missed_signature": "signed, system saw no signature",
    "false_signature": "unsigned, system saw a signature",
}
MAX_EXAMPLES = 3


def field_outcome(kind: FieldType, pred: Optional[FieldResult], truth: FieldResult) -> str:
    if kind == FieldType.CHECKBOX:
        t, p = bool(truth.checked), bool(pred.checked) if pred else False
        return OK if t == p else ("false_tick" if p else "missed_tick")
    if kind == FieldType.SIGNATURE:
        t = truth.status != FieldStatus.MISSING
        p = pred is not None and pred.status != FieldStatus.MISSING
        return OK if t == p else ("false_signature" if p else "missed_signature")
    p_status = pred.status if pred else FieldStatus.MISSING
    if truth.status == FieldStatus.MISSING:
        return OK if p_status == FieldStatus.MISSING else "false_content"
    if truth.status == FieldStatus.UNREADABLE:
        return OK if p_status == FieldStatus.UNREADABLE else "unreadable_not_flagged"
    if p_status == FieldStatus.MISSING:
        return "lost_missing"
    if p_status == FieldStatus.UNREADABLE:
        return "lost_unreadable"
    return OK if _field_text(pred) == _field_text(truth) else "misread"


@dataclass
class Analysis:
    n_forms: int = 0
    outcome_totals: Counter = field(default_factory=Counter)
    per_field: Dict[str, Counter] = field(default_factory=dict)
    examples: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    rules: Dict[str, Counter] = field(default_factory=dict)
    field_rows: List[dict] = field(default_factory=list)
    issue_rows: List[dict] = field(default_factory=list)
    worst_forms: List[Tuple[str, int, int]] = field(default_factory=list)   # (form id, field errors, issue errors)

    @property
    def n_fields(self) -> int:
        return sum(self.outcome_totals.values())

    @property
    def n_field_errors(self) -> int:
        return self.n_fields - self.outcome_totals[OK]

    @property
    def n_issue_errors(self) -> int:
        return len(self.issue_rows)

    @property
    def n_issue_errors_upstream(self) -> int:
        return sum(1 for r in self.issue_rows if r["cause"] == "field_read_error")


def analyse(pairs: Sequence[Tuple[LabelledForm, SystemOutput]], kinds: Mapping[str, FieldType]) -> Analysis:
    a = Analysis()
    per_form: Dict[str, List[int]] = {}
    for form, out in pairs:
        a.n_forms += 1
        pred = {f.name: f for f in out.fields}
        bad_fields = set()
        for t in form.fields:
            p = pred.get(t.name)
            kind = kinds.get(t.name, FieldType.TEXT)
            oc = field_outcome(kind, p, t)
            a.outcome_totals[oc] += 1
            a.per_field.setdefault(t.name, Counter())[oc] += 1
            if oc != OK:
                bad_fields.add(t.name)
            truth_txt, pred_txt = _field_text(t), (_field_text(p) if p else "")
            a.field_rows.append({"form": form.form_id, "field": t.name, "outcome": oc,
                                 "truth_status": t.status.value, "pred_status": (p.status.value if p else "absent"),
                                 "truth": truth_txt if kind not in (FieldType.CHECKBOX, FieldType.SIGNATURE) else
                                 (t.checked if kind == FieldType.CHECKBOX else t.status.value),
                                 "pred": pred_txt if kind not in (FieldType.CHECKBOX, FieldType.SIGNATURE) else
                                 ((p.checked if p else None) if kind == FieldType.CHECKBOX else (p.status.value if p else "absent")),
                                 "ocr_confidence": p.ocr_confidence if p else None})
            if oc == "misread" and len(a.examples[t.name]) < MAX_EXAMPLES:
                a.examples[t.name].append({"form": form.form_id, "truth": truth_txt, "read": pred_txt,
                                           "cer": round(char_error_rate(pred_txt, truth_txt), 2),
                                           "confidence": p.ocr_confidence if p else None})

        want = {issue_key(i): i for i in form.issues}
        got = {issue_key(i): i for i in out.issues}
        n_issue_err = 0
        for key in sorted(set(want) | set(got)):
            rule = (want.get(key) or got.get(key)).id
            c = a.rules.setdefault(rule, Counter())
            if key in want and key in got:
                c["tp"] += 1
                continue
            kind_ = "missed" if key in want else "false_alarm"
            c["fn" if key in want else "fp"] += 1
            issue = want.get(key) or got.get(key)
            wrong = [f for f in issue.fields if f in bad_fields]
            cause = "field_read_error" if wrong else "rule_logic"
            if wrong:
                c["fn_upstream" if kind_ == "missed" else "fp_upstream"] += 1
            a.issue_rows.append({"form": form.form_id, "rule": rule, "fields": ",".join(issue.fields),
                                 "kind": kind_, "cause": cause, "wrongly_read_fields": ",".join(wrong)})
            n_issue_err += 1
        per_form[form.form_id] = [len(bad_fields), n_issue_err]
    a.worst_forms = sorted(((f, v[0], v[1]) for f, v in per_form.items() if v[0] or v[1]),
                           key=lambda x: (-(x[1] + x[2]), x[0]))
    return a


def _pct(x: int, n: int) -> str:
    return "n/a" if not n else f"{100 * x / n:.1f}%"


def format_markdown(a: Analysis, labels: Mapping[str, str], top_forms: int = 8) -> str:
    L = [f"## Where the errors come from ({a.n_forms} forms, {a.n_fields} fields)", "",
         f"{a.n_field_errors} field errors ({_pct(a.n_field_errors, a.n_fields)} of fields) and "
         f"{a.n_issue_errors} wrong issues (missed or false alarms).", "",
         "| Field outcome | Count | Share of fields | Meaning |", "| --- | --- | --- | --- |"]
    for oc in OUTCOMES:
        n = a.outcome_totals[oc]
        if n or oc == OK:
            L.append(f"| {oc} | {n} | {_pct(n, a.n_fields)} | {OUTCOME_HELP[oc]} |")
    if a.n_issue_errors:
        up = a.n_issue_errors_upstream
        L += ["", f"**{up} of {a.n_issue_errors} wrong issues ({_pct(up, a.n_issue_errors)}) trace back to a field "
                  f"that was read wrongly** (fix the vision/OCR side). The other "
                  f"{a.n_issue_errors - up} have all their fields read correctly (look at the rule itself)."]

    L += ["", "### Per field", "", "| Field | n | Correct | Main problem |", "| --- | --- | --- | --- |"]
    rows = []
    for name, c in a.per_field.items():
        n = sum(c.values())
        bad = [(k, v) for k, v in c.items() if k != OK]
        main = max(bad, key=lambda kv: kv[1]) if bad else None
        rows.append((c[OK] / n, name, n, main))
    for acc, name, n, main in sorted(rows):
        L.append(f"| {labels.get(name, name)} | {n} | {100 * acc:.0f}% | "
                 + (f"{main[0]} ({main[1]})" if main else "none") + " |")

    if a.rules:
        L += ["", "### Issues by rule", "",
              "| Rule | Found | Missed | False alarms | Errors caused by a misread field |", "| --- | --- | --- | --- | --- |"]
        for rule, c in sorted(a.rules.items(), key=lambda kv: -(kv[1]["fn"] + kv[1]["fp"])):
            L.append(f"| {rule} | {c['tp']} | {c['fn']} | {c['fp']} | {c['fn_upstream'] + c['fp_upstream']} of {c['fn'] + c['fp']} |")

    ex = [(n, e) for n, es in a.examples.items() for e in es]
    if ex:
        L += ["", "### Examples of misread text", "", "| Form | Field | Truth | Read | CER | OCR confidence |",
              "| --- | --- | --- | --- | --- | --- |"]
        for n, e in ex[:15]:
            L.append(f"| {e['form']} | {labels.get(n, n)} | {e['truth']} | {e['read']} | {e['cer']} | {e['confidence']} |")

    if a.worst_forms:
        L += ["", f"### Forms with the most errors (top {min(top_forms, len(a.worst_forms))})", "",
              "| Form | Field errors | Wrong issues |", "| --- | --- | --- |"]
        for f, fe, ie in a.worst_forms[:top_forms]:
            L.append(f"| {f} | {fe} | {ie} |")
    return "\n".join(L) + "\n"
