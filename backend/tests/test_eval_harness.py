"""Tests for formsense/eval/harness.py."""
from __future__ import annotations

import json
import random

import pytest

from formsense.eval.harness import compare_systems, evaluate_dataset, format_results, load_labels, strip_ground_truth
from formsense.rules.audit import audit
from formsense.schema import FieldResult, FieldStatus as S
from form_factory import TODAY, corrupted_form


@pytest.fixture
def label_dir(tmp_path, mini_tpl):
    from formsense.rules import reference_validator as ref
    tpl_d = json.loads(open("tests/fixtures/mini_template.json").read())
    rng = random.Random(21)
    d = tmp_path / "labels"; d.mkdir()
    for i in range(30):
        fields = corrupted_form(mini_tpl, rng, rng.choice([0, 0, 1, 2, 3]))
        issues = [{"id": t.value, "type": "format", "severity": "error", "fields": sorted(ids),
                  "message": "", "evidence": []} for t, ids in ref.validate(fields, TODAY, tpl_d)]
        doc = {"form_id": f"t_{i:03d}", "reference_date": TODAY.isoformat(),
               "fields": [json.loads(f.model_dump_json()) for f in fields], "issues": issues}
        (d / f"t_{i:03d}.json").write_text(json.dumps(doc))
    return d


def test_load_labels(label_dir):
    forms = load_labels(label_dir)
    assert len(forms) == 30
    assert all(f.reference_date == TODAY for f in forms)


def test_strip_ground_truth_removes_value_and_status_keeps_raw_text():
    f = FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text="35202-1234567-1", value="35202-1234567-1", status=S.OK)
    stripped = strip_ground_truth([f])[0]
    assert stripped.raw_text == "35202-1234567-1" and stripped.value is None and stripped.status == S.OK


def test_full_pipeline_system_recovers_ground_truth_on_clean_input(mini_tpl, label_dir):
    from formsense.eval.harness import full_pipeline_system
    res = evaluate_dataset(label_dir, full_pipeline_system(mini_tpl))
    assert res.n_forms == 30
    assert res.field_extraction.accuracy == pytest.approx(1.0)
    assert res.issue_detection.f1 == pytest.approx(1.0)


def test_evaluate_dataset_respects_limit(mini_tpl, label_dir):
    from formsense.eval.harness import full_pipeline_system
    res = evaluate_dataset(label_dir, full_pipeline_system(mini_tpl), limit=5)
    assert res.n_forms == 5


def test_a_deliberately_worse_system_scores_worse(mini_tpl, label_dir):
    """A system that reports no issues at all should score far below the real engine."""
    class NullOutput:
        def __init__(self, fields):
            self.fields = fields
            self.issues = []

    def do_nothing_system(fields):
        return NullOutput(fields)

    from formsense.eval.harness import full_pipeline_system
    results = compare_systems(label_dir, {"nothing": do_nothing_system, "full": full_pipeline_system(mini_tpl)})
    assert results["nothing"].issue_detection.recall < results["full"].issue_detection.recall
    assert results["full"].issue_detection.f1 == pytest.approx(1.0)


def test_format_results_runs_on_single_and_multi(mini_tpl, label_dir):
    from formsense.eval.harness import full_pipeline_system
    single = evaluate_dataset(label_dir, full_pipeline_system(mini_tpl))
    text = format_results(single)
    assert "system" in text
    multi = compare_systems(label_dir, {"full": full_pipeline_system(mini_tpl)})
    text2 = format_results(multi)
    assert "full" in text2


def test_system_can_opt_into_receiving_todays_reference_date(mini_tpl, label_dir):
    seen = []

    class Out:
        def __init__(self):
            self.fields, self.issues = [], []

    def system_with_today(fields, today):
        seen.append(today)
        return Out()

    evaluate_dataset(label_dir, system_with_today)
    assert len(seen) == 30 and all(d == TODAY for d in seen)
