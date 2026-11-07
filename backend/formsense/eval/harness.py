"""Runs a system over a labelled dataset and reports the metrics from the brief.

    from formsense.eval.harness import evaluate_dataset, compare_systems, full_pipeline_system

    results = evaluate_dataset("data/synthetic/test/labels", full_pipeline_system(tpl))
    print(format_results(results))

    compare_systems("data/synthetic/test/labels",
                    {"baseline": baseline_system(...), "full": full_pipeline_system(tpl)})

A "system" is any callable ``(fields: list[FieldResult]) -> SystemOutput`` where
``SystemOutput`` has ``.fields`` (list[FieldResult]) and ``.issues`` (list[Issue]) -
``rules.audit.AuditResult`` already has this shape, which is what ``full_pipeline_system``
wraps. Wire your own baseline the same way: anything with those two attributes works, no
need to import this module's types.

Ground truth input is read from label JSON files shaped like the generator's output
(``fields``, ``issues``, optionally ``reference_date``). The vision side's OCR output is
simulated by taking the ground truth fields' ``raw_text`` as if it were a perfect OCR
read; to test against lower-quality OCR, corrupt ``raw_text``/``ocr_confidence`` before
calling the system (see ``formsense.eval.ocr_noise`` if you add synthetic noise later,
or pass real Ayesha-pipeline output directly - any list of FieldResult works).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Union

from formsense.eval.metrics import (
    ConfusionMatrix, FieldExtractionScore, PRF, score_field_extraction, score_issue_detection,
    score_missing_field_detection, score_missing_vs_unreadable,
)
from formsense.nlp.template_info import TemplateInfo
from formsense.rules.audit import audit
from formsense.rules.engine import RuleEngine
from formsense.schema import FieldResult, Issue


class SystemOutput(Protocol):
    fields: List[FieldResult]
    issues: List[Issue]


SystemFn = Callable[[Sequence[FieldResult]], SystemOutput]


@dataclass
class LabelledForm:
    form_id: str
    fields: List[FieldResult]
    issues: List[Issue]
    reference_date: Optional[dt.date] = None


def load_labels(label_dir: Union[str, Path]) -> List[LabelledForm]:
    out: List[LabelledForm] = []
    for p in sorted(Path(label_dir).glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        ref = dt.date.fromisoformat(doc["reference_date"]) if "reference_date" in doc else None
        out.append(LabelledForm(
            form_id=doc.get("form_id", p.stem),
            fields=[FieldResult(**d) for d in doc["fields"]],
            issues=[Issue(**d) for d in doc["issues"]],
            reference_date=ref,
        ))
    return out


def full_pipeline_system(
    template: Union[TemplateInfo, Mapping, str, Path, None] = None,
    engine: Optional[RuleEngine] = None,
    today: Optional[dt.date] = None,
) -> SystemFn:
    """A system backed by this project's own extract + rules (formsense.rules.audit).

    ``today`` fixes the audit date for every form; leave it None to use each form's own
    reference_date, which evaluate_dataset supplies automatically when available.
    """
    tpl = TemplateInfo.coerce(template)
    eng = engine or RuleEngine(template=tpl)

    def run(fields: Sequence[FieldResult], _today: Optional[dt.date] = None):
        return audit(fields, tpl, _today or today, engine=eng)
    return run


def strip_ground_truth(fields: Sequence[FieldResult]) -> List[FieldResult]:
    """Simulate raw system input from ground truth: keep raw_text, ink_ratio and
    checked (what a vision pipeline would hand over), drop value and status (what
    extraction is supposed to recompute), so a system is scored honestly on its own work
    rather than being handed the answer.
    """
    out = []
    for f in fields:
        changes = {"value": None, "status": f.status.__class__.OK}
        out.append(f.model_copy(update=changes) if hasattr(f, "model_copy") else f.copy(update=changes))
    return out


@dataclass
class DatasetResult:
    n_forms: int
    field_extraction: FieldExtractionScore
    missing_detection: PRF
    issue_detection: PRF
    issue_detection_by_type: Dict[str, PRF]
    missing_vs_unreadable: ConfusionMatrix

    def as_dict(self) -> dict:
        return {
            "n_forms": self.n_forms,
            "field_extraction": self.field_extraction.as_dict(),
            "missing_field_detection": self.missing_detection.as_dict(),
            "issue_detection": self.issue_detection.as_dict(),
            "issue_detection_by_type": {k: v.as_dict() for k, v in self.issue_detection_by_type.items()},
            "missing_vs_unreadable": self.missing_vs_unreadable.as_dict(),
        }


_ISSUE_TYPES = ("missing", "unreadable", "format", "inconsistency", "selection")


def evaluate_dataset(
    label_dir: Union[str, Path],
    system: SystemFn,
    use_ground_truth_as_input: bool = True,
    limit: Optional[int] = None,
) -> DatasetResult:
    """Run ``system`` over every label in ``label_dir`` and aggregate the brief's metrics.

    use_ground_truth_as_input=True (the default) feeds the system raw_text/ink_ratio from
    ground truth with value/status stripped (see strip_ground_truth) - use this to
    evaluate the language side (nlp/ + rules/) in isolation, independent of real OCR
    quality. Set it False to pass real vision-pipeline FieldResults in yourself by
    pre-building ``forms`` with your own loader instead of this convenience path.
    """
    forms = load_labels(label_dir)
    if limit:
        forms = forms[:limit]
    fe = FieldExtractionScore()
    missing_prf = PRF()
    issue_prf = PRF()
    issue_prf_by_type: Dict[str, PRF] = {t: PRF() for t in _ISSUE_TYPES}
    cm_counts: Dict = {}

    for form in forms:
        sys_input = strip_ground_truth(form.fields) if use_ground_truth_as_input else form.fields
        out = system(sys_input, form.reference_date) if _accepts_today(system) else system(sys_input)

        f_score = score_field_extraction(out.fields, form.fields)
        fe.n += f_score.n; fe.exact_matches += f_score.exact_matches; fe.total_cer += f_score.total_cer
        for name, per in f_score.per_field.items():
            agg = fe.per_field.setdefault(name, FieldExtractionScore())
            agg.n += per.n; agg.exact_matches += per.exact_matches; agg.total_cer += per.total_cer

        missing_prf = missing_prf + score_missing_field_detection(out.fields, form.fields)
        issue_prf = issue_prf + score_issue_detection(out.issues, form.issues)
        for t in _ISSUE_TYPES:
            issue_prf_by_type[t] = issue_prf_by_type[t] + score_issue_detection(out.issues, form.issues, types=[t])

        cm = score_missing_vs_unreadable(out.fields, form.fields)
        for k, v in cm.counts.items():
            cm_counts[k] = cm_counts.get(k, 0) + v

    labels = ("missing", "unreadable", "other") if any(k[1] == "other" for k in cm_counts) else ("missing", "unreadable")
    return DatasetResult(
        n_forms=len(forms), field_extraction=fe, missing_detection=missing_prf,
        issue_detection=issue_prf, issue_detection_by_type=issue_prf_by_type,
        missing_vs_unreadable=ConfusionMatrix(labels=labels, counts=cm_counts),
    )


def _accepts_today(system: SystemFn) -> bool:
    import inspect
    try:
        return len(inspect.signature(system).parameters) >= 2
    except (TypeError, ValueError):
        return False


def compare_systems(
    label_dir: Union[str, Path], systems: Mapping[str, SystemFn], **kwargs
) -> Dict[str, DatasetResult]:
    return {name: evaluate_dataset(label_dir, fn, **kwargs) for name, fn in systems.items()}


def format_results(results: Union[DatasetResult, Mapping[str, DatasetResult]]) -> str:
    """A small text table, good enough for a terminal or to paste into slides."""
    if isinstance(results, DatasetResult):
        results = {"system": results}
    lines = [f"{'system':<14} {'n':>5} {'field acc':>10} {'CER':>7} "
             f"{'miss P/R/F1':>16} {'issue P/R/F1':>16}"]
    for name, r in results.items():
        fe, mp, ip = r.field_extraction, r.missing_detection, r.issue_detection
        lines.append(
            f"{name:<14} {r.n_forms:>5} {fe.accuracy:>10.3f} {fe.mean_cer:>7.3f} "
            f"{mp.precision:>5.2f}/{mp.recall:.2f}/{mp.f1:.2f} "
            f"{ip.precision:>5.2f}/{ip.recall:.2f}/{ip.f1:.2f}"
        )
    return "\n".join(lines)
