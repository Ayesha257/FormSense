"""AuditReport -> PDF checklist, using ReportLab.

    from formsense.report.pdf_report import write_pdf_report
    write_pdf_report(report, "out/dev_0007.pdf", template=tpl)

Layout: a title, the plain English explanation if there is one, then a per field
checklist (one line per field, a tick or a cross, the value or the status), then a table
of issues (severity, the field labels involved, the message). ``template`` is optional
and is only used to show field labels instead of internal names; without it the raw
field names are used.

PII: the PDF shows whatever is in ``report.fields``, same as the JSON report. If the
report is meant to leave the system (emailed, shown to someone other than the applicant),
mask it first with ``formsense.llm.pii_mask.mask_fields`` before calling this, the same
way the LLM prompt is masked. This module does not mask anything itself, since the
applicant's own copy of their own report should show their real CNIC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from formsense.nlp.template_info import TemplateInfo
from formsense.schema import AuditReport, FieldStatus, Severity

_SEVERITY_COLOR = {Severity.ERROR: colors.HexColor("#b00020"), Severity.WARNING: colors.HexColor("#8a6d00"),
                    Severity.INFO: colors.HexColor("#2e5aac")}
_STATUS_MARK = {FieldStatus.OK: "OK", FieldStatus.MISSING: "MISSING", FieldStatus.UNREADABLE: "UNREADABLE",
                FieldStatus.INVALID: "INVALID"}


def _labels(template) -> Mapping[str, str]:
    if template is None:
        return {}
    tpl = TemplateInfo.coerce(template)
    return {name: info.label for name, info in tpl.fields.items()}


def build_pdf(report: AuditReport, template=None) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    labels = _labels(template)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=4 * mm)
    body = styles["BodyText"]

    story = [Paragraph(f"Form Audit Report: {report.form_id}", h1)]
    if report.explanation:
        story += [Paragraph("Summary", styles["Heading2"])]
        for line in report.explanation.split("\n"):
            if line.strip():
                story.append(Paragraph(line, body))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Fields", styles["Heading2"]))
    field_rows = [["Field", "Status", "Value"]]
    for f in report.fields:
        shown = f.value or f.raw_text or ""
        if f.status == FieldStatus.OK and not shown and f.checked is None:
            continue  # checkbox-less empty optional fields with nothing to show, skip clutter
        if f.checked is not None:
            shown = "checked" if f.checked else "unchecked"
        field_rows.append([labels.get(f.name, f.name), _STATUS_MARK[f.status], shown])
    field_table = Table(field_rows, colWidths=[55 * mm, 30 * mm, 85 * mm], repeatRows=1)
    field_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [field_table, Spacer(1, 6 * mm)]

    story.append(Paragraph("Issues", styles["Heading2"]))
    if not report.issues:
        story.append(Paragraph("No issues were found.", body))
    else:
        issue_rows = [["Severity", "Field(s)", "Message"]]
        for i in sorted(report.issues, key=lambda x: {"error": 0, "warning": 1, "info": 2}[x.severity.value]):
            names = ", ".join(labels.get(n, n) for n in i.fields)
            issue_rows.append([i.severity.value.upper(), names, i.message])
        issue_table = Table(issue_rows, colWidths=[22 * mm, 55 * mm, 93 * mm], repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for row_i, i in enumerate(sorted(report.issues, key=lambda x: {"error": 0, "warning": 1, "info": 2}[x.severity.value]), start=1):
            style.append(("TEXTCOLOR", (0, row_i), (0, row_i), _SEVERITY_COLOR[i.severity]))
        issue_table.setStyle(TableStyle(style))
        story.append(issue_table)

    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"FormSense Audit {report.form_id}",
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    doc.build(story)
    return buf.getvalue()


def write_pdf_report(report: AuditReport, path: Union[str, Path], template=None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_pdf(report, template))
    return p
