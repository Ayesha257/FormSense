"""Error breakdown, threshold sweep, simulated OCR, harness refactor, evaluate_all and doctor scripts."""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.eval import failure_analysis as EA  # noqa: E402
from formsense.eval import threshold as TH  # noqa: E402
from formsense.eval.harness import (LabelledForm, aggregate_results, evaluate_dataset,  # noqa: E402
                                    full_pipeline_system, load_labels)
from formsense.eval.simulated_ocr import SimulatedOCR  # noqa: E402
from formsense.nlp.template_info import TemplateInfo  # noqa: E402
from formsense.schema import FieldResult, FieldStatus, FieldType, Issue, IssueType, Severity  # noqa: E402

T, S, M = FieldType.TEXT, FieldType.SIGNATURE, FieldType.CHECKBOX


def fr(name="full_name", status=FieldStatus.OK, text="Ali Raza", value=None, checked=None, conf=0.95):
    return FieldResult(name=name, bbox=(0, 0, 1, 1), status=status, raw_text=text, value=value,
                       checked=checked, ocr_confidence=conf)


# ---------------- field outcomes ----------------

@pytest.mark.parametrize("kind,pred,truth,want", [
    (T, fr(text="Ali Raza"), fr(text="Ali Raza"), "ok"),
    (T, fr(text="Ali Rasa"), fr(text="Ali Raza"), "misread"),
    (T, fr(status=FieldStatus.MISSING, text=""), fr(text="Ali Raza"), "lost_missing"),
    (T, fr(status=FieldStatus.UNREADABLE, text="Ali"), fr(text="Ali Raza"), "lost_unreadable"),
    (T, fr(text="x"), fr(status=FieldStatus.MISSING, text=""), "false_content"),
    (T, fr(status=FieldStatus.MISSING, text=""), fr(status=FieldStatus.MISSING, text=""), "ok"),
    (T, fr(text="Ali Raza"), fr(status=FieldStatus.UNREADABLE, text="Ali Raza"), "unreadable_not_flagged"),
    (T, fr(status=FieldStatus.UNREADABLE, text="?"), fr(status=FieldStatus.UNREADABLE, text="Ali"), "ok"),
    (T, None, fr(text="Ali Raza"), "lost_missing"),
    (M, fr(checked=True), fr(checked=True), "ok"),
    (M, fr(checked=False), fr(checked=True), "missed_tick"),
    (M, fr(checked=True), fr(checked=False), "false_tick"),
    (S, fr(status=FieldStatus.OK), fr(status=FieldStatus.OK), "ok"),
    (S, fr(status=FieldStatus.MISSING), fr(status=FieldStatus.OK), "missed_signature"),
    (S, fr(status=FieldStatus.OK), fr(status=FieldStatus.MISSING), "false_signature"),
])
def test_field_outcome(kind, pred, truth, want):
    assert EA.field_outcome(kind, pred, truth) == want


def test_outcome_compares_normalised_value_when_present():
    assert EA.field_outcome(T, fr(text="13/10/2026", value="2026-10-13"),
                            fr(text="13/10/2026", value="2026-10-13")) == "ok"


# ---------------- issue attribution ----------------

def issue(rid, fields, t=IssueType.FORMAT):
    return Issue(id=rid, type=t, severity=Severity.ERROR, fields=fields, message="m")


class Out:
    def __init__(self, fields, issues):
        self.fields, self.issues = fields, issues


def test_wrong_issues_are_traced_to_misread_fields_or_to_the_rule():
    truth = LabelledForm("f1", [fr("cnic", text="35202-1234567-1"), fr("email", text="a@b.com")],
                         [issue("invalid_email", ["email"])])
    out = Out([fr("cnic", text="35202-1234567-7"), fr("email", text="a@b.com")],     # cnic misread
              [issue("invalid_cnic", ["cnic"])])                                      # false alarm on the misread field
    a = EA.analyse([(truth, out)], {"cnic": T, "email": T})
    rows = {(r["rule"], r["kind"]): r for r in a.issue_rows}
    assert rows[("invalid_cnic", "false_alarm")]["cause"] == "field_read_error"
    assert rows[("invalid_cnic", "false_alarm")]["wrongly_read_fields"] == "cnic"
    assert rows[("invalid_email", "missed")]["cause"] == "rule_logic"                 # email was read right
    assert a.n_issue_errors == 2 and a.n_issue_errors_upstream == 1
    assert a.rules["invalid_cnic"]["fp_upstream"] == 1 and a.rules["invalid_email"]["fn"] == 1


def test_analysis_totals_examples_and_worst_forms():
    good = LabelledForm("g", [fr("full_name", text="Ali")], [])
    bad = LabelledForm("b", [fr("full_name", text="Ayesha")], [])
    a = EA.analyse([(good, Out([fr("full_name", text="Ali")], [])), (bad, Out([fr("full_name", text="Ayasha")], []))],
                   {"full_name": T})
    assert a.n_forms == 2 and a.n_fields == 2 and a.outcome_totals["ok"] == 1 and a.outcome_totals["misread"] == 1
    assert a.examples["full_name"][0]["truth"] == "Ayesha" and a.examples["full_name"][0]["read"] == "Ayasha"
    assert a.worst_forms[0][0] == "b" and len(a.field_rows) == 2


def test_markdown_report_sections():
    truth = LabelledForm("f1", [fr("full_name", text="Ali")], [issue("invalid_email", ["full_name"])])
    a = EA.analyse([(truth, Out([fr("full_name", text="Alx")], []))], {"full_name": T})
    md = EA.format_markdown(a, {"full_name": "Full Name"})
    for part in ("Where the errors come from", "misread", "Per field", "Issues by rule", "Examples of misread text",
                 "trace back to a field"):
        assert part in md


# ---------------- harness refactor ----------------

@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    out = tmp_path_factory.mktemp("forms")
    r = subprocess.run([sys.executable, "-m", "formsense.generator.generate", "--count", "12", "--split", "test",
                        "--seed", "77", "--out", str(out)], cwd=BACKEND, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_aggregate_results_equals_evaluate_dataset(labels):
    tpl = TemplateInfo.load(BACKEND / "data" / "templates" / "admission_form.json")
    system = full_pipeline_system(tpl)
    direct = evaluate_dataset(labels / "labels", system)
    from formsense.eval.harness import strip_ground_truth
    pairs = [(f, system(strip_ground_truth(f.fields), f.reference_date)) for f in load_labels(labels / "labels")]
    same = lambda r: json.dumps(r.as_dict(), sort_keys=True)     # NaN-safe: NaN != NaN in a dict compare  # noqa: E731
    assert same(aggregate_results(pairs)) == same(direct)
    assert direct.n_forms == 12


# ---------------- threshold sweep ----------------

def test_sweep_responds_to_the_cutoff():
    tpl = BACKEND / "data" / "templates" / "admission_form.json"
    raw = [fr("full_name", text="Ali Raza", conf=0.70), fr("father_name", text="Raza Khan", conf=0.95)]
    for f in raw:
        f.ink_ratio = 0.2
    truth = LabelledForm("f", [fr("full_name", text="Ali Raza"), fr("father_name", text="Raza Khan")], [],
                         dt.date(2026, 11, 7))
    rows = TH.sweep([(truth, raw)], tpl, thresholds=(0.6, 0.8))
    low, high = rows
    assert low["fields_flagged_unreadable"] <= high["fields_flagged_unreadable"]
    assert high["unreadable_false_alarms"] >= low["unreadable_false_alarms"]
    assert TH.best(rows)["cutoff"] in (0.6, 0.8) and "best" in TH.format_markdown(rows)


def test_best_prefers_the_lower_cutoff_on_a_tie_and_skips_nan():
    rows = [{"cutoff": 0.5, "issue_f1": 0.7}, {"cutoff": 0.7, "issue_f1": 0.7}, {"cutoff": 0.9, "issue_f1": float("nan")}]
    assert TH.best(rows)["cutoff"] == 0.5


def test_one_shared_unreadable_cutoff():
    from formsense.baseline import ocr_regex
    from formsense.nlp.extract import ExtractionConfig
    from formsense.thresholds import OCR_UNREADABLE_BELOW
    from formsense.vision import pipeline
    assert pipeline.UNREADABLE_BELOW == ocr_regex.UNREADABLE_BELOW == ExtractionConfig().ocr_conf_min == OCR_UNREADABLE_BELOW


# ---------------- simulated OCR ----------------

def test_simulated_ocr_hands_out_texts_in_order_with_deterministic_noise():
    a, b = SimulatedOCR(["one", "two", "three"], noise=0.0), SimulatedOCR(["one", "two", "three"], noise=0.0)
    assert [a.read(None).text for _ in range(3)] == ["one", "two", "three"] and a.read(None).text == ""
    assert all(a_.confidence == 0.95 for a_ in [b.read(None)])
    n1, n2 = SimulatedOCR(["abcdef"] * 40, noise=0.5, seed=3), SimulatedOCR(["abcdef"] * 40, noise=0.5, seed=3)
    r1, r2 = [n1.read(None) for _ in range(40)], [n2.read(None) for _ in range(40)]
    assert r1 == r2 and any(r.text != "abcdef" for r in r1) and any(r.confidence < 0.6 for r in r1)


# ---------------- scripts ----------------

def run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=BACKEND, capture_output=True, text=True, timeout=timeout)


def test_evaluate_all_dry_run_end_to_end(labels, tmp_path):
    out = tmp_path / "eval"
    r = run("scripts/evaluate_all.py", "--dir", str(labels), "--limit", "8", "--dry-run", "--sweep-threshold",
            "--out", str(out))
    assert r.returncode == 0, r.stderr + r.stdout
    d = Path(str(out) + "_dryrun")
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "DRY RUN" in md and "Ceiling (perfect text)" in md and "FormSense (images)" in md
    assert "Text-field accuracy" in md
    assert "Where the errors come from" in md and "OCR confidence cutoff" in md
    s = json.loads((d / "summary.json").read_text())
    assert s["dry_run"] and s["forms"] == 8 and len(s["threshold_sweep"]) == 13
    assert s["systems"]["ceiling"]["field_extraction"]["exact_match_accuracy"] > 0.95
    assert (d / "field_errors_formsense.csv").exists() and any((d / "predictions" / "formsense").glob("*.json"))
    # re-scoring from saved readings needs no OCR and gives the same numbers
    again = run("scripts/evaluate_all.py", "--dir", str(labels), "--limit", "8", "--systems", "formsense", "--reuse",
                "--out", str(out) + "_dryrun")
    assert again.returncode == 0, again.stderr + again.stdout


def test_evaluate_all_refuses_a_folder_without_labels(tmp_path):
    r = run("scripts/evaluate_all.py", "--dir", str(tmp_path), "--dry-run")
    assert r.returncode != 0 and "No labelled forms" in (r.stderr + r.stdout)


def test_doctor_passes_the_wiring_and_retrieval_checks_offline():
    r = run("scripts/doctor.py", "--fake-embeddings", timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "RAG -> explainer: the retrieved guideline reaches the LLM prompt" in out
    assert "RAG retrieval, issues from the rules" in out and "[ FAIL ]" not in out


def test_saved_readings_are_scored_without_ocr_and_match_the_run_that_made_them(labels, tmp_path):
    """--readings takes the full_<form>.json files that formsense.eval.run_comparison writes."""
    out = tmp_path / "eval"
    first = run("scripts/evaluate_all.py", "--dir", str(labels), "--limit", "6", "--dry-run", "--out", str(out))
    assert first.returncode == 0, first.stderr + first.stdout
    d = Path(str(out) + "_dryrun")
    saved = tmp_path / "saved"
    saved.mkdir()
    for p in (d / "predictions" / "formsense").glob("*.json"):
        (saved / f"full_{p.stem}.json").write_text(p.read_text())
    out2 = tmp_path / "eval2"
    r = run("scripts/evaluate_all.py", "--dir", str(labels), "--limit", "6", "--systems", "formsense",
            "--readings", str(saved), "--out", str(out2))
    assert r.returncode == 0, r.stderr + r.stdout
    a = json.loads((d / "summary.json").read_text())["systems"]["formsense"]
    b = json.loads((out2 / "summary.json").read_text())["systems"]["formsense"]
    assert a["field_extraction"] == b["field_extraction"] and a["issue_detection"] == b["issue_detection"]
    md = (out2 / "summary.md").read_text(encoding="utf-8")
    assert "Seconds per form" in md and "| n/a |" in md          # no timing is invented for saved readings
