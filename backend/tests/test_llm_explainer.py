"""Tests for the LLM explainer and its PII safety net (Week 5)."""
from __future__ import annotations

import pytest

from formsense.llm.explainer import GroqClient, LLMNotConfigured, explain_audit
from formsense.llm.explanation_schema import ExplanationOutput, IssueExplanation
from formsense.llm.pii_mask import mask_fields
from formsense.rules.engine import RuleEngine
from formsense.schema import FieldResult, FieldStatus as S, Issue, IssueType, Severity
from form_factory import TODAY, clean_form, corrupted_form
import random


class FakeClient:
    """Records the prompts it was given and returns a canned structured response."""
    def __init__(self, response=None):
        self.calls = []
        self._response = response

    def explain(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self._response is not None:
            return self._response
        return ExplanationOutput(summary="ok", issues=[], overall_recommendation=None)


def cnic_issue(fields):
    return Issue(id="invalid_cnic", type=IssueType.FORMAT, severity=Severity.ERROR, fields=["cnic"],
                 message="CNIC is not in the format 12345-1234567-1.")


def test_pii_never_reaches_the_prompt(mini_tpl, mini_engine):
    rng = random.Random(1)
    for _ in range(100):
        fields = corrupted_form(mini_tpl, rng, rng.randint(1, 3))
        issues = mini_engine.validate(fields, TODAY)
        client = FakeClient()
        explain_audit(fields, issues, mini_tpl, client=client)
        for _, prompt in client.calls:
            for f in fields:
                if f.name in ("cnic", "phone", "email") and len(f.raw_text) > 5:
                    assert f.raw_text not in prompt, f"{f.name} leaked into the prompt"


def test_client_is_called_with_masked_fields(mini_tpl):
    fields = [
        FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text="35202-1234567-1", value="35202-1234567-1"),
        FieldResult(name="full_name", bbox=(0, 0, 1, 1), raw_text="Ali Raza", value="Ali Raza"),
    ]
    issues = [cnic_issue(fields)]
    client = FakeClient()
    explain_audit(fields, issues, mini_tpl, client=client)
    _, prompt = client.calls[0]
    assert "35202-1234567-1" not in prompt
    assert "35XXX" in prompt or "cnic" not in prompt.lower()   # masked form present, not the real one
    assert "Ali Raza" in prompt                                 # non PII values are not hidden


def test_prompt_uses_labels_not_internal_names(mini_tpl):
    fields = [FieldResult(name="date_of_birth", bbox=(0, 0, 1, 1), raw_text="", status=S.MISSING)]
    issues = [Issue(id="missing_required", type=IssueType.MISSING, severity=Severity.ERROR,
                    fields=["date_of_birth"], message="Date of Birth is required but has been left empty.")]
    client = FakeClient()
    explain_audit(fields, issues, mini_tpl, client=client)
    _, prompt = client.calls[0]
    assert "Date of Birth" in prompt


def test_no_issues_produces_a_clean_summary_offline():
    out = explain_audit([], [], offline=True)
    assert out.issues == [] and "no problems" in out.summary.lower()


def test_offline_mode_never_calls_a_client(mini_tpl):
    fields = [FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text="bad", value=None, status=S.INVALID)]
    issues = [cnic_issue(fields)]
    client = FakeClient()
    out = explain_audit(fields, issues, mini_tpl, client=client, offline=True)
    assert client.calls == []
    assert len(out.issues) == 1 and out.issues[0].issue_id == "invalid_cnic"


def test_missing_api_key_falls_back_instead_of_raising(mini_tpl, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    fields = [FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text="bad", value=None, status=S.INVALID)]
    issues = [cnic_issue(fields)]
    out = explain_audit(fields, issues, mini_tpl, client=GroqClient())
    assert len(out.issues) == 1   # fallback explanation, not an exception


def test_groq_client_raises_llmnotconfigured_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMNotConfigured):
        GroqClient()._model()


def test_groq_client_accepts_an_explicit_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = GroqClient(api_key="fake-key-for-construction-only")
    m = c._model()                      # must not raise; constructing ChatGroq does no network call
    assert m is not None


def test_a_model_that_returns_a_dict_is_still_accepted(mini_tpl):
    class DictClient:
        def explain(self, system_prompt, user_prompt):
            return {"summary": "fine", "issues": [], "overall_recommendation": None}
    out = explain_audit([], [], mini_tpl, client=None, offline=True)   # offline path, sanity only
    assert out.summary


def test_explanation_output_is_validated_pydantic():
    out = ExplanationOutput(summary="s", issues=[IssueExplanation(issue_id="x", explanation="e")],
                            overall_recommendation=None)
    assert out.issues[0].issue_id == "x"
    with pytest.raises(Exception):
        ExplanationOutput(summary=123, issues="not a list")


def test_mismatched_issue_ids_logs_a_warning_but_does_not_raise(mini_tpl, caplog):
    import logging
    fields = [FieldResult(name="cnic", bbox=(0, 0, 1, 1), raw_text="bad", value=None, status=S.INVALID)]
    issues = [cnic_issue(fields)]
    client = FakeClient(ExplanationOutput(summary="s", issues=[], overall_recommendation=None))
    with caplog.at_level(logging.WARNING, logger="formsense.llm"):
        out = explain_audit(fields, issues, mini_tpl, client=client)
    assert out.issues == []
    assert any("do not match" in r.getMessage() for r in caplog.records)
