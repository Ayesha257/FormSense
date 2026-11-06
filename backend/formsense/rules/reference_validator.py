"""Plain Python reference implementation of every rule.
Purpose: (1) proves the synthetic ground truth is self consistent (see tests), (2) seed for the YAML rule engine.
Input is a list of FieldResult, so it works for ground truth and for real pipeline output alike."""
from __future__ import annotations
import datetime as dt, json, pathlib, re
from typing import Dict, List, Optional
from rapidfuzz import fuzz
from formsense.schema import FieldResult, FieldStatus as S
from formsense.rules.codes import Code as T

ROOT = pathlib.Path(__file__).resolve().parents[2]
CNIC_RE = re.compile(r"^\d{5}-\d{7}-\d$")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^03\d{2}-\d{7}$")
PLACEHOLDER_RE = re.compile(r"^(n/?a|nil|none|null|x+|[-._\s]+)$", re.I)
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"]
NAME_THRESHOLD = 80      # rapidfuzz token_sort_ratio
PCT_TOLERANCE = 1.0      # percentage points
MIN_AGE = 16


def load_template(name="admission_form") -> dict:
    return json.loads((ROOT / "data" / "templates" / f"{name}.json").read_text())


def parse_date(raw: str) -> Optional[dt.date]:
    for f in DATE_FORMATS:
        try:
            return dt.datetime.strptime(raw.strip(), f).date()
        except ValueError:
            pass
    return None


def to_num(raw: str) -> Optional[float]:
    try:
        return float(raw.strip().rstrip("%").strip())
    except ValueError:
        return None


def normalise(ftype: str, raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    if ftype == "date":
        d = parse_date(raw); return d.isoformat() if d else None
    if ftype == "number":
        return raw.rstrip("%").strip() if to_num(raw) is not None else None
    return raw


def names_match(a: str, b: str) -> bool:
    n = lambda s: re.sub(r"[^a-z ]", "", s.lower())
    return fuzz.token_sort_ratio(n(a), n(b)) >= NAME_THRESHOLD


def age_on(dob: dt.date, today: dt.date) -> int:
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def validate(fields: List[FieldResult], today: dt.date, tpl: dict = None):
    """Return a list of (IssueType, frozenset(field ids))."""
    tpl = tpl or load_template()
    F: Dict[str, FieldResult] = {f.name: f for f in fields}
    spec = {f["name"]: f for f in tpl["fields"]}
    out = []
    add = lambda t, *ids: out.append((t, frozenset(ids)))
    bad = set()  # fields that already carry a field level problem: skip cross checks on them

    def txt(n):  # text if the field is usable, else None
        f = F[n]
        return None if f.status in (S.MISSING, S.UNREADABLE) or n in bad else f.raw_text.strip()

    # 1. required / unreadable / placeholder / format
    for n, sp in spec.items():
        if sp["type"] in ("checkbox",):
            continue
        f = F[n]
        if f.status == S.MISSING and sp["required"]:
            add(T.MISSING_REQUIRED, n); bad.add(n); continue
        if f.status == S.UNREADABLE:
            add(T.UNREADABLE_FIELD, n); bad.add(n); continue
        if sp["type"] == "signature":
            continue
        raw = f.raw_text.strip()
        if PLACEHOLDER_RE.match(raw):
            add(T.PLACEHOLDER_VALUE, n); bad.add(n); continue
        if n == "cnic" and not CNIC_RE.match(raw): add(T.INVALID_CNIC, n); bad.add(n)
        elif n == "email" and not EMAIL_RE.match(raw): add(T.INVALID_EMAIL, n); bad.add(n)
        elif n == "phone" and not PHONE_RE.match(raw): add(T.INVALID_PHONE, n); bad.add(n)
        elif sp["type"] == "date" and parse_date(raw) is None: add(T.INVALID_DATE, n); bad.add(n)
        elif sp["kind"] == "number" and (to_num(raw) is None): add(T.INVALID_NUMBER, n); bad.add(n)
        elif n.endswith("_year") and not re.match(r"^\d{4}$", raw): add(T.INVALID_NUMBER, n); bad.add(n)
    # 2. groups
    ticked = {}
    for g, gs in tpl["groups"].items():
        ids = [i for i, s in spec.items() if s.get("group") == g]
        ticked[g] = [i for i in ids if F[i].checked is True]
        if not ticked[g]: add(T.NO_SELECTION, *ids)
        elif len(ticked[g]) > 1: add(T.MULTIPLE_SELECTIONS, *ids)
    # 3. cross field
    dob = parse_date(txt("date_of_birth")) if txt("date_of_birth") else None
    decl = parse_date(txt("declaration_date")) if txt("declaration_date") else None
    if dob and age_on(dob, today) < MIN_AGE: add(T.UNDERAGE, "date_of_birth")
    if decl and decl > today: add(T.DECLARATION_DATE_FUTURE, "declaration_date")
    if dob and decl and decl < dob: add(T.DECLARATION_BEFORE_DOB, "date_of_birth", "declaration_date")
    c = txt("cnic")
    if c and CNIC_RE.match(c) and len(ticked["gender"]) == 1:
        male = int(c[-1]) % 2 == 1
        if male != (ticked["gender"][0] == "gender_male"): add(T.CNIC_GENDER_MISMATCH, "cnic", "gender_male", "gender_female")
    my, iy = txt("matric_year"), txt("inter_year")
    if my and iy and int(my) >= int(iy): add(T.YEAR_ORDER, "matric_year", "inter_year")
    for k in ("matric", "inter"):
        tot, obt, pct = txt(f"{k}_total"), txt(f"{k}_obtained"), txt(f"{k}_percentage")
        if tot and obt and to_num(tot) and to_num(tot) > 0:
            if to_num(obt) > to_num(tot): add(T.MARKS_EXCEED_TOTAL, f"{k}_total", f"{k}_obtained")
            elif pct and abs(to_num(obt) / to_num(tot) * 100 - to_num(pct)) > PCT_TOLERANCE:
                add(T.PERCENTAGE_MISMATCH, f"{k}_total", f"{k}_obtained", f"{k}_percentage")
    a, b = txt("full_name"), txt("declaration_name")
    if a and b and not names_match(a, b): add(T.NAME_MISMATCH, "full_name", "declaration_name")
    return out
