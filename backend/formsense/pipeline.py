"""The whole FormSense chain for one form page, in one call.

    page image -> vision (alignment, ink, OCR) -> rules (extract + validate)
               -> guideline passage for every issue (RAG)

    from formsense.pipeline import FormSense
    fs = FormSense()                          # loads template, OCR and guidelines lazily
    result = fs.run(page_bgr)
    result.issues        # list[Issue] from the rules
    result.fields        # FieldResult with final status (INVALID where a format rule fired)
    result.citations     # {issue key: GuidelineCitation}
    result.report()      # AuditReport (schema.py) for the LLM explainer and the report generator
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from formsense.schema import AuditReport, FieldResult, GuidelineCitation, Issue
from formsense.vision.pipeline import VisionOutput, analyze_page
from formsense.vision.template import DEFAULT_TEMPLATE, load_template


def issue_key(issue: Issue) -> str:
    """Several issues can share a rule id; the key also names the fields."""
    return f"{issue.id}:{','.join(issue.fields)}"


@dataclass
class FormResult:
    form_id: str
    vision: VisionOutput
    fields: list[FieldResult]
    issues: list[Issue]
    notes: dict[str, list[str]] = field(default_factory=dict)
    citations: dict[str, GuidelineCitation] = field(default_factory=dict)

    def report(self, explanation: str | None = None) -> AuditReport:
        return AuditReport(form_id=self.form_id, fields=self.fields, issues=self.issues,
                           explanation=explanation, citations=list(self.citations.values()))


class FormSense:
    def __init__(self, template_path: Path | str = DEFAULT_TEMPLATE, use_ocr: bool = True,
                 use_trocr: bool = True, use_rag: bool = True, ocr=None, retriever=None):
        self.template_path = Path(template_path)
        self.template, self.blank = load_template(self.template_path)
        self.use_ocr, self.use_trocr, self.use_rag = use_ocr, use_trocr, use_rag
        self._ocr, self._retriever = ocr, retriever
        self._labels = None

    @property
    def ocr(self):
        if self._ocr is None and self.use_ocr:
            from formsense.vision.ocr import FieldOCR
            self._ocr = FieldOCR(use_trocr=self.use_trocr)
        return self._ocr

    @property
    def retriever(self):
        if self._retriever is None and self.use_rag:
            from formsense.rag import GuidelineRetriever
            self._retriever = GuidelineRetriever().ensure()
        return self._retriever

    @property
    def labels(self) -> dict[str, str]:
        if self._labels is None:
            from formsense.rag.retriever import field_labels
            self._labels = field_labels(self.template_path)
        return self._labels

    def run(self, page: np.ndarray, today: dt.date | None = None, form_id: str = "form") -> FormResult:
        from formsense.rules.audit import audit
        vision = analyze_page(page, self.template, self.blank, ocr=self.ocr)
        res = audit(vision.fields, template=self.template_path, today=today or dt.date.today())
        out = FormResult(form_id, vision, res.fields, res.issues, res.notes)
        if self.retriever is not None:
            for issue in res.issues:
                cites = self.retriever.for_issue(issue, self.labels)
                if cites:
                    out.citations[issue_key(issue)] = cites[0]
        return out
