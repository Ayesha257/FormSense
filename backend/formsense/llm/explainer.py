"""Turns an audit (fields + issues) into a plain English explanation.

    from formsense.llm.explainer import explain_audit, GroqClient

    out = explain_audit(result.fields, result.issues, template)
    out.summary              # str
    out.issues                # list[IssueExplanation]
    out.overall_recommendation

Pluggable client
-----------------
The actual model call goes through an ``LLMClient``: anything with an
``explain(system_prompt, user_prompt) -> ExplanationOutput`` method. The default is
``GroqClient``, built lazily so importing this module never requires network access or
an API key. Tests pass a fake client (see ``tests/test_llm_explainer.py``); a Streamlit
app or a notebook can pass the same default client used here.

PII safety
----------
``explain_audit`` always masks CNIC, phone and email (via ``formsense.llm.pii_mask``)
before building the prompt, and calls ``assert_no_pii`` on the finished prompt as a last
check before any client is invoked. If masking is somehow bypassed, the call raises
instead of sending the value.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, Union

from formsense.llm.explanation_schema import ExplanationOutput, IssueExplanation
from formsense.llm.pii_mask import assert_no_pii, mask_fields
from formsense.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from formsense.nlp.template_info import TemplateInfo
from formsense.schema import FieldResult, Issue

log = logging.getLogger("formsense.llm")

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMClient(Protocol):
    def explain(self, system_prompt: str, user_prompt: str) -> ExplanationOutput: ...


class LLMNotConfigured(RuntimeError):
    """Raised when GroqClient is used without an API key and no key was given."""


@dataclass
class GroqClient:
    """Default client: LangChain's ChatGroq with structured output.

    Nothing here runs at import time or at construction time beyond reading an env var;
    the network call and the langchain_groq import both happen inside ``explain``, so a
    module that only type-checks against ``LLMClient`` never needs the package installed.
    """
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.2

    def _model(self):
        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMNotConfigured(
                "no Groq API key: set GROQ_API_KEY in your environment (.env) or pass "
                "GroqClient(api_key=...)"
            )
        from langchain_groq import ChatGroq  # imported here: keeps this module importable without the package
        return ChatGroq(model=self.model, api_key=key, temperature=self.temperature)

    def explain(self, system_prompt: str, user_prompt: str) -> ExplanationOutput:
        llm = self._model().with_structured_output(ExplanationOutput)
        result = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        if isinstance(result, ExplanationOutput):
            return result
        return ExplanationOutput.model_validate(result)  # some integrations return a dict


def _labels(template: Union[TemplateInfo, Mapping, str, None]) -> Mapping[str, str]:
    """Field name -> label, or {} if no template is available.

    Falling back to {} (rather than raising) means explain_audit still works when there
    are no issues to explain and no template was given: prompts.py falls back to the raw
    field name for anything not in this map.
    """
    try:
        tpl = TemplateInfo.coerce(template)
    except (OSError, FileNotFoundError):
        log.debug("no template available for labels; using raw field names")
        return {}
    return {name: info.label for name, info in tpl.fields.items()}


def _fallback_output(issues: Sequence[Issue]) -> ExplanationOutput:
    """A plain, template based explanation used when no LLM client is available.

    This keeps the rest of the pipeline (report generation, the demo) usable without a
    Groq key. It is not a substitute for the real explanations the evaluation Week
    compares against; it exists so the system degrades instead of breaking.
    """
    if not issues:
        return ExplanationOutput(
            summary="The form looks correctly filled in. No problems were found.",
            issues=[], overall_recommendation=None,
        )
    by_sev = sorted(issues, key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.severity.value])
    summary = (f"{len(issues)} issue(s) were found on this form, including "
               f"{sum(1 for i in issues if i.severity.value == 'error')} that must be fixed.")
    out_issues = [IssueExplanation(issue_id=i.id, explanation=i.message) for i in issues]
    return ExplanationOutput(
        summary=summary, issues=out_issues,
        overall_recommendation=f"Please correct the problem with {', '.join(by_sev[0].fields)} first.",
    )


def explain_audit(
    fields: Sequence[FieldResult],
    issues: Sequence[Issue],
    template: Union[TemplateInfo, Mapping, str, None] = None,
    client: Optional[LLMClient] = None,
    guideline_context: Optional[Sequence[str]] = None,
    extra_pii_fields: Sequence[str] = (),
    offline: bool = False,
) -> ExplanationOutput:
    """Explain one audit. Masks PII, builds the prompt, calls the client.

    offline=True (or no client available and none configured) skips the LLM entirely and
    returns a plain templated explanation instead of raising, so a demo or a batch job
    does not hard fail just because a key is missing.
    """
    labels = _labels(template)
    masked = mask_fields(fields, extra_pii_fields)
    prompt = build_user_prompt(masked.fields, issues, labels, guideline_context)
    assert_no_pii(prompt, fields, extra_pii_fields)   # checked against the ORIGINAL values

    if offline:
        return _fallback_output(issues)

    client = client or GroqClient()
    try:
        out = client.explain(SYSTEM_PROMPT, prompt)
    except LLMNotConfigured:
        log.warning("no LLM configured; returning a templated explanation instead")
        return _fallback_output(issues)
    assert_no_pii(out.summary + " " + " ".join(e.explanation for e in out.issues), fields, extra_pii_fields)
    _align_issue_order(out, issues)
    return out


def _align_issue_order(out: ExplanationOutput, issues: Sequence[Issue]) -> None:
    """Warn (do not fail) if the model dropped or reordered issues; downstream code should
    still be able to zip ``issues`` with ``out.issues`` by id."""
    got = {e.issue_id for e in out.issues}
    want = {i.id for i in issues}
    if got != want:
        log.warning("LLM explanation issue ids do not match the input exactly: missing=%s extra=%s",
                    want - got, got - want)
