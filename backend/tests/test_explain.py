"""Glue between pipeline and language side, and the explainer's Groq client (fake ChatGroq, no network)."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense import explain as E  # noqa: E402
from formsense import groq_util  # noqa: E402
from formsense.llm.explanation_schema import ExplanationOutput, IssueExplanation  # noqa: E402
from formsense.schema import (AuditReport, FieldResult, FieldStatus, GuidelineCitation, Issue,  # noqa: E402
                              IssueType, Severity)

CNIC = "35202-1234567-1"


def cite(text, n=1):
    return GuidelineCitation(issue_id=f"r{n}", text=text, source=f"guidelines.md#s{n}")


def report(citations=()):
    fields = [FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text=CNIC, value=CNIC),
              FieldResult(name="signature", bbox=(0, 0, 1, 1), status=FieldStatus.MISSING)]
    issues = [Issue(id="signature_missing", type=IssueType.MISSING, severity=Severity.ERROR,
                    fields=["signature"], message="Signature is missing")]
    return AuditReport(form_id="f1", fields=fields, issues=issues, citations=list(citations))


class Recorder:
    def __init__(self):
        self.prompts = []

    def explain(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return ExplanationOutput(summary="s", issues=[IssueExplanation(issue_id="signature_missing", explanation="e")],
                                 overall_recommendation="r")


# ---------------- glue ----------------

def test_guideline_context_dedupes_limits_and_formats():
    cs = [cite("A", 1), cite("A", 2), cite("B", 3), cite("C", 4), cite("D", 5), cite("E", 6)]
    ctx = E.guideline_context(cs)
    assert ctx == ["guidelines.md#s1: A", "guidelines.md#s3: B", "guidelines.md#s4: C", "guidelines.md#s5: D"]
    assert E.guideline_context([]) == []


def test_rag_arm_includes_guidelines_and_llm_alone_arm_does_not():
    rep = report([cite("Unsigned forms are returned to the applicant.")])
    with_rag, alone = Recorder(), Recorder()
    E.explain_report(rep, use_rag=True, client=with_rag)
    E.explain_report(rep, use_rag=False, client=alone)
    assert "Unsigned forms are returned" in with_rag.prompts[0]
    assert "Unsigned forms are returned" not in alone.prompts[0]
    # everything else in the prompt is identical, so the arms differ only in the guidelines
    assert with_rag.prompts[0].split("Relevant excerpts")[0].strip() == alone.prompts[0].split("Write the summary")[0].strip()


def test_no_citations_means_no_guideline_section_even_with_rag_on():
    rec = Recorder()
    E.explain_report(report(), use_rag=True, client=rec)
    assert "Relevant excerpts" not in rec.prompts[0]


def test_pii_never_reaches_the_client():
    rec = Recorder()
    E.explain_report(report([cite("CNIC must have 13 digits.")]), client=rec)
    assert CNIC not in rec.prompts[0]


def test_without_a_key_it_falls_back_to_a_template_and_says_so(monkeypatch):
    monkeypatch.setattr(groq_util, "load_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert E.is_offline() and not E.has_llm_key()
    out = E.explain_report(report())
    assert "issue(s) were found" in out.summary


def test_a_given_client_is_never_offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert not E.is_offline(Recorder())


def test_build_full_report_and_pdf():
    rep = report([cite("x")])
    result = types.SimpleNamespace(form_id="f1", fields=rep.fields, issues=rep.issues,
                                   citations={"signature_missing:signature": rep.citations[0]})
    out = Recorder().explain("", "")
    full = E.build_full_report(result, out)
    assert full.form_id == "f1" and full.citations == rep.citations
    assert full.explanation.splitlines()[0] == "s" and "- e" in full.explanation and E.explanation_text(out) == full.explanation
    assert E.build_full_report(result, None).explanation is None
    assert E.pdf_bytes(full).startswith(b"%PDF")


# ---------------- retired model handling in the glue ----------------

class Boom(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


class Failing:
    def __init__(self, exc):
        self.exc = exc

    def explain(self, system_prompt, user_prompt):
        raise self.exc


def test_retired_model_becomes_an_actionable_error(monkeypatch):
    monkeypatch.setattr(groq_util, "load_env", lambda: None)
    monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
    with pytest.raises(E.LLMModelUnavailable) as e:
        E.explain_report(report(), client=Failing(Boom(404, "Error code: 404 model_not_found, model does not exist")))
    assert "GROQ_LLM_MODEL" in str(e.value) and groq_util.DEFAULT_TEXT_MODEL in str(e.value)


def test_other_errors_pass_through_unchanged():
    with pytest.raises(Boom):
        E.explain_report(report(), client=Failing(Boom(401, "invalid api key")))


def test_text_model_env_names(monkeypatch):
    monkeypatch.setattr(groq_util, "load_env", lambda: None)
    for v in ("GROQ_LLM_MODEL", "GROQ_TEXT_MODEL"):
        monkeypatch.delenv(v, raising=False)
    assert groq_util.text_model() == groq_util.DEFAULT_TEXT_MODEL
    monkeypatch.setenv("GROQ_TEXT_MODEL", "alias/model")
    assert groq_util.text_model() == "alias/model"
    monkeypatch.setenv("GROQ_LLM_MODEL", "main/model")       # the name llm/explainer.py reads wins
    assert groq_util.text_model() == "main/model" and groq_util.text_model("arg/model") == "arg/model"
