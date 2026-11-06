"""Builds ground truth style FieldResults from a template, and corrupts them.

It mirrors what the generator writes (value = text for strings, ISO for dates, stripped
number for numbers) without importing the generator, so the rule engine can be tested
on its own and compared against the reference validator.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Callable, Dict, List, Tuple

from formsense.nlp.template_info import TemplateInfo
from formsense.rules import reference_validator as ref
from formsense.schema import FieldResult, FieldStatus

TODAY = dt.date(2026, 11, 6)


def gt_value(info, raw: str):
    """What the generator stores in FieldResult.value (same as reference_validator.normalise)."""
    ftype = "date" if info.type == "date" else ("number" if info.kind == "number" else info.type)
    return ref.normalise(ftype, raw)


def _fr(info, raw="", checked=None, status=FieldStatus.OK, ink=0.05, conf=None) -> FieldResult:
    value = None if info.type in ("checkbox", "signature") else gt_value(info, raw)
    return FieldResult(name=info.name, bbox=info.bbox, raw_text=raw, value=value, ocr_confidence=conf,
                       ink_ratio=ink, checked=checked, status=status)


def clean_form(tpl: TemplateInfo, rng: random.Random) -> Dict[str, FieldResult]:
    """One internally consistent, error free form."""
    male = rng.random() < 0.5
    cnic = f"{rng.randint(10000, 99999)}-{rng.randint(1000000, 9999999)}-{rng.choice([1, 3, 5, 7, 9] if male else [0, 2, 4, 6, 8])}"
    full_name = rng.choice(["Ali Raza", "Ayesha Khan", "Hamza Iqbal", "Fatima Noor", "Usman Tariq Butt"])
    dob = dt.date(rng.randint(2003, 2008), rng.randint(1, 12), rng.randint(1, 28))
    decl = dt.date(2026, rng.randint(1, 10), rng.randint(1, 28))
    fmt = rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"])
    program = rng.choice([n for n, i in tpl.fields.items() if i.group == "program"] or [None])
    totals = {k: rng.choice([1000, 1100, 1200]) for k in ("matric", "inter")}
    obtained = {k: int(totals[k] * rng.uniform(0.55, 0.95)) for k in totals}
    years = {"matric": rng.randint(2019, 2021), "inter": rng.randint(2022, 2024)}
    out: Dict[str, FieldResult] = {}
    for name, info in tpl.fields.items():
        if info.type == "checkbox":
            if info.group == "gender":
                tick = name == ("gender_male" if male else "gender_female")
            else:
                tick = name == program
            out[name] = _fr(info, checked=tick, ink=0.2 if tick else 0.0)
        elif info.type == "signature":
            out[name] = _fr(info, ink=0.06)
        elif info.type == "date":
            d = dob if "birth" in name else decl
            out[name] = _fr(info, d.strftime(fmt))
        else:
            raw = {
                "full_name": full_name, "father_name": "Muhammad Aslam", "cnic": cnic,
                "email": "student.name@example.com", "phone": f"03{rng.randint(0, 4)}{rng.randint(0, 9)}-{rng.randint(1000000, 9999999)}",
                "address": "House 12, Street 4, Model Town, Lahore", "declaration_name": full_name,
            }.get(name)
            if raw is None:
                prefix = name.split("_")[0]
                if name.endswith("_year"):
                    raw = str(years.get(prefix, 2020))
                elif name.endswith("_total"):
                    raw = str(totals.get(prefix, 1000))
                elif name.endswith("_obtained"):
                    raw = str(obtained.get(prefix, 800))
                elif name.endswith("_percentage"):
                    raw = f"{obtained[prefix] / totals[prefix] * 100:.1f}" if prefix in obtained else "80.0"
                else:
                    raw = "Sample text"
            out[name] = _fr(info, raw)
    return out


# ---------------------------------------------------------------- corruptions
Mutator = Callable[[Dict[str, FieldResult], TemplateInfo, random.Random], None]


def _set(forms, tpl, name, raw):
    if name in tpl.fields and tpl.fields[name].type != "checkbox":
        forms[name] = _fr(tpl.fields[name], raw)


def m_blank_text(f, t, r):
    names = [n for n, i in t.fields.items() if i.type in ("text", "date") and i.required]
    n = r.choice(names)
    f[n] = _fr(t.fields[n], "", status=FieldStatus.MISSING, ink=0.0)


def m_blank_signature(f, t, r):
    n = next((n for n, i in t.fields.items() if i.type == "signature"), None)
    if n:
        f[n] = _fr(t.fields[n], "", status=FieldStatus.MISSING, ink=0.0)


def m_unreadable(f, t, r):
    names = [n for n, i in t.fields.items() if i.type in ("text", "date")]
    n = r.choice(names)
    f[n] = _fr(t.fields[n], "x#~s", status=FieldStatus.UNREADABLE, ink=0.04, conf=0.3)


def m_placeholder(f, t, r):
    names = [n for n, i in t.fields.items() if i.type in ("text", "date")]
    n = r.choice(names)
    _set(f, t, n, r.choice(["N/A", "-", "nil", "xxxx"]))


def m_bad_cnic(f, t, r):
    c = f["cnic"].raw_text
    if not ref.CNIC_RE.match(c):      # already corrupted by another mutator; do not stack
        return
    _set(f, t, "cnic", r.choice([c.replace("-", ""), c[:-1], c + "1", c.replace("0", "O", 1) if "0" in c else c[:-1]]))


def m_bad_email(f, t, r):
    _set(f, t, "email", r.choice(["student.name@@example", "student.name.example.com", "a@b"]))


def m_bad_phone(f, t, r):
    _set(f, t, "phone", r.choice(["0400-1234567", "03001234567", "0300-123456"]))


def m_bad_date(f, t, r):
    n = r.choice([n for n, i in t.fields.items() if i.type == "date"])
    _set(f, t, n, r.choice(["31/02/2005", "45/13/2004", "00/05/2005", "soon"]))


def m_bad_number(f, t, r):
    names = [n for n, i in t.fields.items() if i.kind == "number" or n.endswith("_year")]
    n = r.choice(names)
    _set(f, t, n, r.choice(["eighty", "8O0", "twenty twenty"]))


def m_underage(f, t, r):
    _set(f, t, "date_of_birth", "01/01/2015")


def m_decl_future(f, t, r):
    _set(f, t, "declaration_date", "01/01/2027")


def m_decl_before_dob(f, t, r):
    _set(f, t, "declaration_date", "01/01/2001")


def m_gender_flip(f, t, r):
    for n in ("gender_male", "gender_female"):
        if n in f:
            f[n] = _fr(t.fields[n], checked=not f[n].checked, ink=0.2 if not f[n].checked else 0.0)


def m_two_ticks(f, t, r):
    names = [n for n, i in t.fields.items() if i.group == "program"]
    for n in names[:2]:
        f[n] = _fr(t.fields[n], checked=True, ink=0.2)


def m_no_ticks(f, t, r):
    names = [n for n, i in t.fields.items() if i.group == "program"]
    for n in names:
        f[n] = _fr(t.fields[n], checked=False, ink=0.0)


def m_year_order(f, t, r):
    _set(f, t, "matric_year", "2023")
    _set(f, t, "inter_year", "2021")


def m_marks_exceed(f, t, r):
    k = r.choice(["matric", "inter"])
    _set(f, t, f"{k}_obtained", "1500")


def m_pct_mismatch(f, t, r):
    k = r.choice(["matric", "inter"])
    _set(f, t, f"{k}_percentage", "50")


def m_name_mismatch(f, t, r):
    _set(f, t, "declaration_name", "Zainab Bukhari")


MUTATORS: List[Mutator] = [
    m_blank_text, m_blank_signature, m_unreadable, m_placeholder, m_bad_cnic, m_bad_email, m_bad_phone,
    m_bad_date, m_bad_number, m_underage, m_decl_future, m_decl_before_dob, m_gender_flip, m_two_ticks,
    m_no_ticks, m_year_order, m_marks_exceed, m_pct_mismatch, m_name_mismatch,
]


def corrupted_form(tpl: TemplateInfo, rng: random.Random, n_errors: int) -> List[FieldResult]:
    forms = clean_form(tpl, rng)
    for m in rng.sample(MUTATORS, k=n_errors):
        try:
            m(forms, tpl, rng)
        except KeyError:
            pass                      # template lacks a field this mutator needs
    return list(forms.values())
