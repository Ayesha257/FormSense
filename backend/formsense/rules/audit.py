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
    ap = argparse.ArgumentParser(description="Audit a JSON list of field readings (or a label file).")
    ap.add_argument("file")
    ap.add_argument("--today")
    ap.add_argument("--no-extract", action="store_true", help="fields already have final status and value")
    args = ap.parse_args(argv)
    doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
    raw = doc["fields"] if isinstance(doc, dict) else doc
    res = audit([FieldResult(**d) for d in raw],
                today=dt.date.fromisoformat(args.today) if args.today else None, extract=not args.no_extract)
    dump = lambda m: m.model_dump(mode="json") if hasattr(m, "model_dump") else json.loads(m.json())
    print(json.dumps({"issues": [dump(i) for i in res.issues], "notes": res.notes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
