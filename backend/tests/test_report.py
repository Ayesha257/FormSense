"""Tests for formsense/report/ (JSON + PDF report generation, Week 6)."""
from __future__ import annotations

import json
import random

from formsense.llm.explainer import explain_audit
from formsense.llm.explanation_schema import ExplanationOutput, IssueExplanation
from formsense.report.build import build_report, explanation_to_text
from formsense.report.json_report import report_to_json, write_json_report
from formsense.report.pdf_report import build_pdf, write_pdf_report
from formsense.rules.audit import audit
from formsense.schema import AuditReport, FieldResult, FieldStatus as S, Issue, IssueType, Severity
from form_factory import TODAY, corrupted_form


def test_explanation_to_text_joins_in_order():
    out = ExplanationOutput(summary="Summary here.", issues=[IssueExplanation(issue_id="x", explanation="Detail.")],
                            overall_recommendation="Fix x.")
    text = explanation_to_text(out)
    assert text.splitlines() == ["Summary here.", "- Detail.", "Fix x."]


def test_explanation_to_text_handles_no_recommendation():
    out = ExplanationOutput(summary="All good.", issues=[], overall_recommendation=None)
    assert explanation_to_text(out) == "All good."


def test_build_report_without_explanation():
    report = build_report("f1", [], [])
    assert report.form_id == "f1" and report.explanation is None and report.citations == []


def test_build_report_with_explanation():
    out = ExplanationOutput(summary="Looks fine.", issues=[], overall_recommendation=None)
    report = build_report("f1", [], [], explanation=out)
    assert report.explanation == "Looks fine."


def test_build_report_is_a_real_audit_report(mini_tpl):
    rng = random.Random(4)
    fields = corrupted_form(mini_tpl, rng, 2)
    res = audit(fields, mini_tpl, TODAY, extract=False)
    exp = explain_audit(res.fields, res.issues, mini_tpl, offline=True)
    report = build_report("f2", res.fields, res.issues, exp)
    assert isinstance(report, AuditReport)
    assert len(report.issues) == len(res.issues)


# ---------- JSON ----------
def test_json_report_round_trips(mini_tpl):
    rng = random.Random(5)
    fields = corrupted_form(mini_tpl, rng, 1)
    res = audit(fields, mini_tpl, TODAY, extract=False)
    report = build_report("f3", res.fields, res.issues)
    text = report_to_json(report)
    back = AuditReport.model_validate(json.loads(text))
    assert back.form_id == "f3" and len(back.issues) == len(res.issues)


def test_write_json_report_creates_parent_dirs(tmp_path, mini_tpl):
    report = build_report("f4", [], [])
    p = write_json_report(report, tmp_path / "nested" / "dir" / "f4.json")
    assert p.exists()
    assert json.loads(p.read_text())["form_id"] == "f4"


# ---------- PDF ----------
def test_build_pdf_is_a_real_pdf(mini_tpl):
    rng = random.Random(6)
    fields = corrupted_form(mini_tpl, rng, 2)
    res = audit(fields, mini_tpl, TODAY, extract=False)
    exp = explain_audit(res.fields, res.issues, mini_tpl, offline=True)
    report = build_report("f5", res.fields, res.issues, exp)
    pdf_bytes = build_pdf(report, template=mini_tpl)
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500


def test_pdf_with_no_issues_does_not_crash(mini_tpl):
    report = build_report("f6", [], [])
    pdf_bytes = build_pdf(report, template=mini_tpl)
    assert pdf_bytes[:4] == b"%PDF"


def test_write_pdf_report_creates_file(tmp_path, mini_tpl):
    report = build_report("f7", [], [])
    p = write_pdf_report(report, tmp_path / "out" / "f7.pdf", template=mini_tpl)
    assert p.exists() and p.read_bytes()[:4] == b"%PDF"


def test_pdf_works_without_a_template(mini_tpl):
    rng = random.Random(7)
    fields = corrupted_form(mini_tpl, rng, 1)
    res = audit(fields, mini_tpl, TODAY, extract=False)
    report = build_report("f8", res.fields, res.issues)
    pdf_bytes = build_pdf(report, template=None)   # falls back to raw field names
    assert pdf_bytes[:4] == b"%PDF"
