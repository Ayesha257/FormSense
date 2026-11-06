"""YAML driven validation engine.

Reads ``rules.yaml`` and turns a list of ``FieldResult`` into a list of ``Issue``.

How a run works
---------------
1. Field level rules, per field in template order. Rules run in YAML order and the first
   *blocking* hit wins: the field is then marked "bad" and hidden from later cross field
   checks. This is what stops one empty box from producing five confusing messages.
2. Selection rules over checkbox groups (nothing ticked, several ticked).
3. Cross field rules (age, date order, CNIC versus gender, marks, names ...). A cross
   field rule silently skips any field that is missing, unreadable or already bad.

Robustness
----------
* A broken or unexpected input never crashes a run: each rule is isolated, errors are
  logged (rule id and field names only, never values) and the rest still run.
  Pass ``raise_errors=True`` in tests to see them.
* Messages are built from templates in the YAML and contain labels, never values.
* The engine reads ``FieldResult.value`` when set, else ``raw_text``. Ground truth and
  the extractor both keep ``value`` equal to the text for strings and ISO for dates.

Adding a rule
-------------
Most new rules need only a YAML entry that reuses an existing ``check``. A new kind of
check is one function registered with ``@field_check`` or ``@cross_check`` below.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import yaml

from formsense.nlp.normalize import (
    DEFAULT_DATE_FORMATS,
    names_match,
    parse_date_strict,
    to_number,
)
from formsense.nlp.template_info import FieldInfo, TemplateInfo
from formsense.schema import FieldResult, FieldStatus, Issue, IssueType, Severity

log = logging.getLogger("formsense.rules")

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")
_SKIP_FORMAT_TYPES = ("signature", "checkbox")
# Parameter keys that hold field names; used by the start up lint.
_FIELD_PARAM_KEYS = {
    "field", "dob", "earlier", "later", "cnic", "a", "b", "male_field", "total", "obtained", "percentage",
}


class RuleConfigError(ValueError):
    """The rules file is malformed."""


# ============================================================ rule specification
@dataclass(frozen=True)
class RuleSpec:
    id: str
    check: str
    type: IssueType
    severity: Severity
    message: str
    description: str = ""
    enabled: bool = True
    blocking: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)


def load_rules(path: Union[str, Path, None] = None) -> Tuple[List[RuleSpec], Dict[str, Any]]:
    p = Path(path) if path else DEFAULT_RULES_PATH
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise RuleConfigError(f"cannot read rules file {p}: {e}") from e
    if not isinstance(doc, dict) or "rules" not in doc:
        raise RuleConfigError(f"{p}: expected a mapping with a 'rules' list")
    settings = dict(doc.get("settings") or {})
    rules: List[RuleSpec] = []
    seen: Set[str] = set()
    for i, raw in enumerate(doc["rules"]):
        where = f"{p} rule #{i + 1} ({raw.get('id', '?')})"
        for key in ("id", "check", "type", "severity", "message"):
            if key not in raw:
                raise RuleConfigError(f"{where}: missing '{key}'")
        if raw["id"] in seen:
            raise RuleConfigError(f"{where}: duplicate rule id")
        seen.add(raw["id"])
        if raw["check"] not in CHECKS:
            raise RuleConfigError(f"{where}: unknown check '{raw['check']}' (known: {sorted(CHECKS)})")
        try:
            rtype = IssueType(raw["type"])
            sev = Severity(raw["severity"])
        except ValueError as e:
            raise RuleConfigError(f"{where}: {e}") from e
        rules.append(RuleSpec(
            id=raw["id"], check=raw["check"], type=rtype, severity=sev, message=str(raw["message"]),
            description=" ".join(str(raw.get("description", "")).split()),
            enabled=bool(raw.get("enabled", True)), blocking=bool(raw.get("blocking", True)),
            params=dict(raw.get("params") or {}),
        ))
    return rules, settings


# ============================================================ evaluation context
class Ctx:
    """Everything a check needs: template, results, today's date, and the 'bad' set."""

    def __init__(self, tpl: TemplateInfo, results: Dict[str, FieldResult], today: dt.date,
                 formats: Sequence[str], flags: Mapping[str, Iterable[str]]):
        self.tpl = tpl
        self.results = results
        self.today = today
        self.formats = formats
        self.flags = flags
        self.bad: Set[str] = set()

    @staticmethod
    def view(f: FieldResult) -> str:
        return (f.value or "").strip() or (f.raw_text or "").strip()

    def text(self, name: str) -> Optional[str]:
        """Usable text of a field, or None if it is absent, missing, unreadable, bad or empty."""
        f = self.results.get(name)
        if f is None or f.status in (FieldStatus.MISSING, FieldStatus.UNREADABLE) or name in self.bad:
            return None
        return self.view(f) or None

    def date(self, name: str) -> Optional[dt.date]:
        t = self.text(name)
        return parse_date_strict(t, self.formats) if t else None

    def ticked(self, group: str) -> List[str]:
        return [n for n in self.tpl.groups.get(group, []) if (self.results.get(n) is not None and self.results[n].checked is True)]


# ============================================================ check registry
FieldCheck = Callable[[Ctx, RuleSpec, FieldInfo, FieldResult], bool]
GroupedCheck = Callable[[Ctx, RuleSpec], List[List[str]]]
CHECKS: Dict[str, Tuple[str, Callable]] = {}      # name -> (scope, function)


def field_check(name: str):
    def deco(fn: FieldCheck):
        CHECKS[name] = ("field", fn)
        return fn
    return deco


def group_check(name: str):
    def deco(fn: GroupedCheck):
        CHECKS[name] = ("group", fn)
        return fn
    return deco


def cross_check(name: str):
    def deco(fn: GroupedCheck):
        CHECKS[name] = ("cross", fn)
        return fn
    return deco


# ---------------------------------------------------------------- field level checks
@field_check("required_missing")
def _required_missing(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    if not info.required:
        return False
    if f.status == FieldStatus.MISSING:
        return True
    if f.status == FieldStatus.UNREADABLE or info.type == "signature":
        return False
    return not ctx.view(f)             # status says OK but nothing was captured


@field_check("unreadable")
def _unreadable(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    return f.status == FieldStatus.UNREADABLE


def _format_text(ctx: Ctx, info: FieldInfo, f: FieldResult) -> Optional[str]:
    """Text to format check, or None when the field should be skipped."""
    if info.type in _SKIP_FORMAT_TYPES:
        return None
    return ctx.view(f) or None         # empty optional fields are not format errors


@field_check("placeholder")
def _placeholder(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    t = _format_text(ctx, info, f)
    return t is not None and re.match(rule.params["pattern"], t, re.I) is not None


@field_check("pattern")
def _pattern(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    if info.name != rule.params["field"]:
        return False
    t = _format_text(ctx, info, f)
    return t is not None and re.match(rule.params["pattern"], t) is None


@field_check("date_parse")
def _date_parse(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    if info.type not in rule.params.get("types", ["date"]):
        return False
    t = _format_text(ctx, info, f)
    return t is not None and parse_date_strict(t, ctx.formats) is None


@field_check("numeric")
def _numeric(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    t = _format_text(ctx, info, f)
    if t is None:
        return False
    if info.kind in rule.params.get("kinds", ["number"]):
        return to_number(t) is None
    suffix = rule.params.get("year_suffix")
    if suffix and info.name.endswith(suffix):
        return re.match(rule.params.get("year_pattern", r"^\d{4}$"), t) is None
    return False


@field_check("external_flag")
def _external_flag(ctx: Ctx, rule: RuleSpec, info: FieldInfo, f: FieldResult) -> bool:
    return rule.params["flag"] in set(ctx.flags.get(info.name, ()))


# ---------------------------------------------------------------- selection checks
@group_check("group_selection")
def _group_selection(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    mode = rule.params["mode"]
    only = rule.params.get("groups")
    hits: List[List[str]] = []
    for g, ids in ctx.tpl.groups.items():
        if only and g not in only:
            continue
        n = len(ctx.ticked(g))
        if (mode == "none" and n == 0) or (mode == "multiple" and n > 1):
            hits.append(list(ids))
    return hits


# ---------------------------------------------------------------- cross field checks
def _age_on(dob: dt.date, today: dt.date) -> int:
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@cross_check("min_age")
def _min_age(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    name = rule.params["dob"]
    d = ctx.date(name)
    return [[name]] if d and _age_on(d, ctx.today) < rule.params["min_age"] else []


@cross_check("not_in_future")
def _not_in_future(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    name = rule.params["field"]
    d = ctx.date(name)
    return [[name]] if d and d > ctx.today else []


@cross_check("date_order")
def _date_order(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    a, b = rule.params["earlier"], rule.params["later"]
    da, db = ctx.date(a), ctx.date(b)
    return [[a, b]] if da and db and db < da else []


@cross_check("cnic_gender")
def _cnic_gender(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    p = rule.params
    c = ctx.text(p["cnic"])
    ticked = ctx.ticked(p["group"])
    if not c or re.match(p["pattern"], c) is None or len(ticked) != 1:
        return []
    is_male = int(c[-1]) % 2 == 1
    if is_male != (ticked[0] == p["male_field"]):
        return [[p["cnic"], *ctx.tpl.groups.get(p["group"], [])]]
    return []


@cross_check("year_order")
def _year_order(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    a, b = rule.params["earlier"], rule.params["later"]
    ta, tb = ctx.text(a), ctx.text(b)
    if ta and tb and ta.isdigit() and tb.isdigit() and int(ta) >= int(tb):
        return [[a, b]]
    return []


def _marks(ctx: Ctx, s: Mapping[str, str]) -> Optional[Tuple[float, float]]:
    tot, obt = to_number(ctx.text(s["total"]) or ""), to_number(ctx.text(s["obtained"]) or "")
    if tot is None or obt is None or tot <= 0:
        return None
    return tot, obt


@cross_check("marks_exceed_total")
def _marks_exceed(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    hits = []
    for s in rule.params["sets"]:
        m = _marks(ctx, s)
        if m and m[1] > m[0]:
            hits.append([s["total"], s["obtained"]])
    return hits


@cross_check("percentage_mismatch")
def _percentage(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    tol = float(rule.params.get("tolerance", 1.0))
    hits = []
    for s in rule.params["sets"]:
        m = _marks(ctx, s)
        if not m or m[1] > m[0]:       # marks_exceed_total already covers that case
            continue
        pct = to_number(ctx.text(s["percentage"]) or "")
        if pct is not None and abs(m[1] / m[0] * 100 - pct) > tol:
            hits.append([s["total"], s["obtained"], s["percentage"]])
    return hits


@cross_check("name_match")
def _name_match(ctx: Ctx, rule: RuleSpec) -> List[List[str]]:
    a, b = rule.params["a"], rule.params["b"]
    ta, tb = ctx.text(a), ctx.text(b)
    if ta and tb and not names_match(ta, tb, rule.params.get("threshold", 80)):
        return [[a, b]]
    return []


# ============================================================ engine
class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class RuleEngine:
    def __init__(self, rules_path: Union[str, Path, None] = None,
                 template: Union[TemplateInfo, Mapping, str, Path, None] = None,
                 raise_errors: bool = False):
        self.tpl = TemplateInfo.coerce(template)
        rules, self.settings = load_rules(rules_path)
        self.rules = [r for r in rules if r.enabled]
        self.formats: Sequence[str] = tuple(self.settings.get("date_formats") or DEFAULT_DATE_FORMATS)
        self.raise_errors = raise_errors
        self.warnings: List[str] = self._lint()
        for w in self.warnings:
            log.warning(w)

    # ---------- public API ----------
    def validate(self, fields: Sequence[FieldResult], today: Optional[dt.date] = None,
                 flags: Optional[Mapping[str, Iterable[str]]] = None, strict: bool = False) -> List[Issue]:
        """Return the issues found in ``fields``.

        today   date used for age and "future" checks (default: the system date).
        flags   optional vision signals per field, e.g. {"address": {"struck_through"}}.
        strict  raise if the template has fields the input lacks (default: treat as MISSING).
        """
        results = self._index(fields, strict)
        ctx = Ctx(self.tpl, results, today or dt.date.today(), self.formats, flags or {})
        issues: List[Issue] = []

        field_rules = [r for r in self.rules if CHECKS[r.check][0] == "field"]
        blocking = [r for r in field_rules if r.blocking]
        advisory = [r for r in field_rules if not r.blocking]
        for info in self.tpl.fields.values():
            if info.type == "checkbox":
                continue
            f = results[info.name]
            for rule in blocking:
                if self._guard(rule, CHECKS[rule.check][1], ctx, rule, info, f):
                    issues.append(self._issue(rule, [info.name], ctx))
                    ctx.bad.add(info.name)
                    break
            for rule in advisory:
                if self._guard(rule, CHECKS[rule.check][1], ctx, rule, info, f):
                    issues.append(self._issue(rule, [info.name], ctx))

        for scope in ("group", "cross"):
            for rule in self.rules:
                s, fn = CHECKS[rule.check]
                if s != scope:
                    continue
                hits = self._guard(rule, fn, ctx, rule) or []
                for names in hits:
                    issues.append(self._issue(rule, names, ctx))
        return issues

    def catalogue(self) -> List[Dict[str, str]]:
        return [{"id": r.id, "type": r.type.value, "severity": r.severity.value,
                 "check": r.check, "description": r.description} for r in self.rules]

    # ---------- internals ----------
    def _index(self, fields: Sequence[FieldResult], strict: bool) -> Dict[str, FieldResult]:
        results = {f.name: f for f in fields}
        unknown = [n for n in results if n not in self.tpl.fields]
        if unknown:
            log.debug("ignoring fields not in the template: %s", unknown)
        absent = [n for n in self.tpl.fields if n not in results]
        if absent:
            if strict:
                raise ValueError(f"input is missing template fields: {absent}")
            log.warning("fields absent from input, treated as MISSING: %s", absent)
            for n in absent:
                results[n] = FieldResult(name=n, bbox=self.tpl.fields[n].bbox, status=FieldStatus.MISSING)
        return results

    def _guard(self, rule: RuleSpec, fn: Callable, *args):
        try:
            return fn(*args)
        except Exception:
            if self.raise_errors:
                raise
            log.exception("rule %s failed and was skipped", rule.id)
            return False

    def _issue(self, rule: RuleSpec, names: Sequence[str], ctx: Ctx) -> Issue:
        scalars = {k: v for k, v in rule.params.items() if isinstance(v, (str, int, float))}
        text = rule.message.format_map(_SafeDict(
            labels=", ".join(self.tpl.label(n) for n in names), fields=", ".join(names), **scalars))
        evidence = []
        for n in names:
            r = ctx.results.get(n)
            bb = r.bbox if r is not None else self.tpl.bbox(n)
            if bb is not None:
                evidence.append(tuple(int(v) for v in bb))
        return Issue(id=rule.id, type=rule.type, severity=rule.severity, fields=list(names),
                     message=text, evidence=evidence)

    def _lint(self) -> List[str]:
        out: List[str] = []

        def walk(rule_id: str, obj: Any, key: Optional[str] = None):
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    walk(rule_id, v, k)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(rule_id, v, key)
            elif isinstance(obj, str) and key in _FIELD_PARAM_KEYS and obj not in self.tpl.fields:
                out.append(f"rule '{rule_id}' refers to field '{obj}' which is not in the template; it will never fire")
            elif isinstance(obj, str) and key == "group" and obj not in self.tpl.groups:
                out.append(f"rule '{rule_id}' refers to group '{obj}' which is not in the template; it will never fire")

        for r in self.rules:
            walk(r.id, dict(r.params))
        return out


# ============================================================ helpers for callers
def mark_invalid(fields: Sequence[FieldResult], issues: Sequence[Issue]) -> List[FieldResult]:
    """Set status INVALID on fields named by a FORMAT issue (MISSING/UNREADABLE are kept)."""
    bad = {n for i in issues if i.type == IssueType.FORMAT and i.severity != Severity.INFO for n in i.fields}
    out = []
    for f in fields:
        if f.name in bad and f.status == FieldStatus.OK:
            changes = {"status": FieldStatus.INVALID}
            out.append(f.model_copy(update=changes) if hasattr(f, "model_copy") else f.copy(update=changes))
        else:
            out.append(f)
    return out


def issue_key(issue: Issue) -> Tuple[str, frozenset]:
    """Comparable identity of an issue: rule id plus the set of fields."""
    return issue.id, frozenset(issue.fields)


# ============================================================ command line
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the FormSense rule engine on a label or fields JSON file.")
    ap.add_argument("file", nargs="?", help="label JSON (with a 'fields' list) or a plain list of fields")
    ap.add_argument("--today", help="YYYY-MM-DD, default today")
    ap.add_argument("--rules", help="alternative rules.yaml")
    ap.add_argument("--template", help="alternative template JSON")
    ap.add_argument("--list", action="store_true", help="print the rule catalogue and exit")
    args = ap.parse_args(argv)

    engine = RuleEngine(args.rules, args.template)
    if args.list:
        for r in engine.catalogue():
            print(f"{r['id']:26} {r['type']:13} {r['severity']:8} {r['description']}")
        return 0
    if not args.file:
        ap.error("give a file, or use --list")
    doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
    raw_fields = doc["fields"] if isinstance(doc, dict) else doc
    fields = [FieldResult(**d) for d in raw_fields]
    today = dt.date.fromisoformat(args.today) if args.today else None
    issues = engine.validate(fields, today)
    print(json.dumps([i.model_dump(mode="json") if hasattr(i, "model_dump") else json.loads(i.json())
                      for i in issues], indent=2))
    if isinstance(doc, dict) and "issues" in doc:
        want = {(d["id"], frozenset(d["fields"])) for d in doc["issues"] if d["id"] != "correction_mark"}
        got = {issue_key(i) for i in issues}
        print(f"\nrecorded issues: {len(want)}  engine issues: {len(got)}  "
              f"{'MATCH' if want == got else 'DIFFERENT'}", file=sys.stderr)
        if want != got:
            print(f"  only in recorded: {sorted(map(str, want - got))}", file=sys.stderr)
            print(f"  only in engine:   {sorted(map(str, got - want))}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
