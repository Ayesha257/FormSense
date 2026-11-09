"""Glue between the pipeline and the language side: explanation, report, PDF.

    from formsense.pipeline import FormSense
    from formsense.explain import explain_report, build_full_report, pdf_bytes

    result = FormSense().run(page)                       # vision -> rules -> guideline per issue
    out = explain_report(result.report())                # LLM explanation, guidelines included (RAG)
    report = build_full_report(result, out)              # AuditReport with explanation + citations
    pdf = pdf_bytes(report)

`explain_report` is the single place where retrieved guidelines (Ayesha's rag/) reach the
explainer (Easha's llm/): the citations on the report become `guideline_context`. Leaving them
out (`use_rag=False`) gives the "LLM alone" arm of the explanation comparison, with everything
else identical, so the two arms differ only in the guidelines.

Without a Groq key the explainer returns its templated fallback instead of raising, so the app,
reports and demos keep working; `is_offline()` tells the UI which kind of text it is showing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from formsense import groq_util
from formsense.llm.explainer import LLMClient, explain_audit
from formsense.llm.explanation_schema import ExplanationOutput
from formsense.report.build import build_report, explanation_to_text
from formsense.report.pdf_report import build_pdf
from formsense.schema import AuditReport, GuidelineCitation
from formsense.vision.template import DEFAULT_TEMPLATE

class LLMModelUnavailable(RuntimeError):
    """Groq no longer serves the configured text model. The message says what to change."""


MAX_CONTEXT = 4   # distinct guideline passages in the prompt: enough to ground it, short enough to stay focused


def has_llm_key() -> bool:
    groq_util.load_env()
    import os
    return bool(os.environ.get("GROQ_API_KEY"))


def is_offline(client: Optional[LLMClient] = None) -> bool:
    """True when explain_report would use the templated fallback (no client given and no key)."""
    return client is None and not has_llm_key()


def guideline_context(citations: Sequence[GuidelineCitation], limit: int = MAX_CONTEXT) -> list[str]:
    """Distinct passages as 'source: text', in the order of the issues they belong to."""
    seen, out = set(), []
    for c in citations:
        if c.text in seen:
            continue
        seen.add(c.text)
        out.append(f"{c.source}: {c.text}")
        if len(out) == limit:
            break
    return out


def explain_report(report: AuditReport, template: Path | str = DEFAULT_TEMPLATE, *, use_rag: bool = True,
                   offline: Optional[bool] = None, client: Optional[LLMClient] = None) -> ExplanationOutput:
    """Plain-English explanation of an audit. PII is masked by the explainer before any model call.

    use_rag     pass the report's guideline citations to the model (False = "LLM alone")
    offline     None = decide from the key; True = templated text, no model call
    """
    ctx = guideline_context(report.citations) if use_rag else None
    try:
        return explain_audit(report.fields, report.issues, template, client=client, guideline_context=ctx or None,
                             offline=is_offline(client) if offline is None else offline)
    except Exception as e:
        if groq_util.is_model_gone(e):      # a retired model: say what to change instead of a raw API error
            raise LLMModelUnavailable(groq_util.model_gone_message(
                None, groq_util.text_model(), "GROQ_LLM_MODEL", groq_util.DEFAULT_TEXT_MODEL)) from e
        raise


def build_full_report(result, explanation: Optional[ExplanationOutput] = None) -> AuditReport:
    """AuditReport (fields, issues, flattened explanation, citations) from a pipeline FormResult."""
    return build_report(result.form_id, result.fields, result.issues, explanation,
                        list(result.citations.values()))


def explanation_text(out: Optional[ExplanationOutput]) -> Optional[str]:
    return None if out is None else explanation_to_text(out)


def pdf_bytes(report: AuditReport, template: Path | str = DEFAULT_TEMPLATE) -> bytes:
    return build_pdf(report, template)
