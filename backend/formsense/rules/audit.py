"""One call from field readings to a finished audit: extract, validate, annotate.

    from formsense.rules.audit import audit
    result = audit(field_results, today=date.today())
    result.fields   # FieldResult with status and value filled in (INVALID where a format rule fired)
    result.issues   # list[Issue]
    result.notes    # {field name: [human readable extraction notes]}

This is the function the pipeline, the LLM explainer and the report generator call.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from formsense.nlp.extract import ExtractionConfig, extract_fields
from formsense.nlp.template_info import TemplateInfo
from formsense.rules.engine import RuleEngine, mark_invalid
from formsense.schema import FieldResult, Issue


@dataclass
class AuditResult:
    fields: List[FieldResult]
    issues: List[Issue]
    notes: Dict[str, List[str]] = field(default_factory=dict)


def audit(
    fields: Sequence[FieldResult],
    template: Union[TemplateInfo, Mapping, str, Path, None] = None,
    today: Optional[dt.date] = None,
    cfg: Optional[ExtractionConfig] = None,
    engine: Optional[RuleEngine] = None,
    flags: Optional[Mapping[str, Iterable[str]]] = None,
    extract: bool = True,
) -> AuditResult:
    """Run the full language side on one form.

    extract=False skips the extraction step; use it for ground truth fields whose status
    and value are already final.
    """
    tpl = TemplateInfo.coerce(template if template is not None else (engine.tpl if engine else None))
    engine = engine or RuleEngine(template=tpl)
    notes: Dict[str, List[str]] = {}
    work = list(fields)
    if extract:
        extractions = extract_fields(work, tpl, cfg or ExtractionConfig())
        work = [e.result for e in extractions]
        notes = {e.result.name: e.notes for e in extractions if e.notes}
    issues = engine.validate(work, today, flags)
    return AuditResult(fields=mark_invalid(work, issues), issues=issues, notes=notes)


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit a JSON list of field readings (or a label file), "
                                              "optionally explain it and write a JSON/PDF report.")
    ap.add_argument("file")
    ap.add_argument("--today")
    ap.add_argument("--no-extract", action="store_true", help="fields already have final status and value")
    ap.add_argument("--template", help="template JSON (defaults to data/templates/admission_form.json)")
    ap.add_argument("--explain", action="store_true", help="also run the LLM explainer")
    ap.add_argument("--offline", action="store_true", help="use the templated explanation, no LLM call")
    ap.add_argument("--form-id", default=None)
    ap.add_argument("--report-out", help="directory to write <form_id>.json and <form_id>.pdf into")
    args = ap.parse_args(argv)
    doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
    raw = doc["fields"] if isinstance(doc, dict) else doc
    form_id = args.form_id or (doc.get("form_id") if isinstance(doc, dict) else None) or Path(args.file).stem

    res = audit([FieldResult(**d) for d in raw], template=args.template,
                today=dt.date.fromisoformat(args.today) if args.today else None, extract=not args.no_extract)
    dump = lambda m: m.model_dump(mode="json") if hasattr(m, "model_dump") else json.loads(m.json())

    explanation = None
    if args.explain or args.report_out:
        from formsense.llm.explainer import explain_audit
        explanation = explain_audit(res.fields, res.issues, args.template, offline=args.offline)

    if args.report_out:
        from formsense.report.build import build_report
        from formsense.report.json_report import write_json_report
        from formsense.report.pdf_report import write_pdf_report
        report = build_report(form_id, res.fields, res.issues, explanation)
        out_dir = Path(args.report_out)
        write_json_report(report, out_dir / f"{form_id}.json")
        write_pdf_report(report, out_dir / f"{form_id}.pdf", template=args.template)
        print(f"wrote {out_dir / f'{form_id}.json'} and {out_dir / f'{form_id}.pdf'}", file=sys.stderr)

    payload = {"issues": [dump(i) for i in res.issues], "notes": res.notes}
    if explanation is not None:
        payload["explanation"] = dump(explanation)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
