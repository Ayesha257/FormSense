"""Assembles the final AuditReport from everything upstream has produced.

    from formsense.report.build import build_report
    report = build_report(form_id="dev_0007", fields=result.fields, issues=result.issues,
                           explanation=explanation.summary, citations=[])

This is a thin, deliberately boring function: all the real decisions (status, issues,
wording) were already made earlier in the pipeline. Its only job is to put them in the
shape ``schema.AuditReport`` expects, with a couple of sensible defaults filled in.
"""
from __future__ import annotations

from typing import Optional, Sequence

from formsense.llm.explanation_schema import ExplanationOutput
from formsense.schema import AuditReport, FieldResult, GuidelineCitation, Issue


def explanation_to_text(out: ExplanationOutput) -> str:
    """Flatten the structured LLM output into the single string AuditReport.explanation
    expects. Order: summary, then one line per issue, then the recommendation."""
    lines = [out.summary]
    for e in out.issues:
        lines.append(f"- {e.explanation}")
    if out.overall_recommendation:
        lines.append(out.overall_recommendation)
    return "\n".join(lines)


def build_report(
    form_id: str,
    fields: Sequence[FieldResult],
    issues: Sequence[Issue],
    explanation: Optional[ExplanationOutput] = None,
    citations: Sequence[GuidelineCitation] = (),
) -> AuditReport:
    return AuditReport(
        form_id=form_id,
        fields=list(fields),
        issues=list(issues),
        explanation=explanation_to_text(explanation) if explanation is not None else None,
        citations=list(citations),
    )
