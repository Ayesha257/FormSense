"""AuditReport -> JSON.

    from formsense.report.json_report import report_to_json, write_json_report
    text = report_to_json(report)
    write_json_report(report, "out/dev_0007.json")

Kept tiny on purpose: AuditReport is already a Pydantic model, so there is very little to
do here beyond a consistent indent and a path helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from formsense.schema import AuditReport


def report_to_json(report: AuditReport, indent: int = 2) -> str:
    return report.model_dump_json(indent=indent)


def write_json_report(report: AuditReport, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_to_json(report), encoding="utf-8")
    return p
