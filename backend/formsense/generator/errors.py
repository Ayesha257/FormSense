"""Error injectors: each one breaks a valid form in ONE specific way and records the expected Issue.
Registry: ERRORS (real mistakes, ground truth issues) and BENIGN (odd but valid, must NOT raise issues,
they measure false positives).  An injector returns the set of field ids it touched, or None if not applicable.
Two injectors never touch the same field (so ground truth stays unambiguous)."""
from __future__ import annotations
import copy, datetime as dt, random
from dataclasses import dataclass, field
from typing import Callable, Dict, List
from formsense.schema import Issue
from formsense.rules.codes import Code as T, COARSE, SEVERITY
from formsense.generator.data import SURNAMES, MALE, FEMALE, DATE_STYLES, add_years, fmt_date, pick_name
from formsense.rules.reference_validator import parse_date, names_match, CNIC_RE, EMAIL_RE, PHONE_RE

REQUIRED_TEXT = ["full_name", "father_name", "cnic", "date_of_birth", "email", "phone", "address", "matric_year", "matric_total",
                 "matric_obtained", "matric_percentage", "inter_year", "inter_total", "inter_obtained", "inter_percentage",
                 "declaration_name", "declaration_date"]
NUMERIC = [f"{k}_{x}" for k in ("matric", "inter") for x in ("year", "total", "obtained", "percentage")]


@dataclass
class FormState:
    text: Dict[str, str]
    ticks: Dict[str, bool]
    signature: str                       # full | faint | speck | none
    style: dict
    today: dt.date
    status: Dict[str, str] = field(default_factory=dict)    # field -> MISSING | UNREADABLE | INVALID
    defects: Dict[str, object] = field(default_factory=dict)  # field -> "illegible" | ("strike", struck_text)
    issues: List[Issue] = field(default_factory=list)
    applied: List[str] = field(default_factory=list)
    benign: List[str] = field(default_factory=list)


def mk(t, fields, msg):
    return Issue(id=t.value, type=COARSE[t], severity=SEVERITY[t], fields=list(fields), message=msg)


def exam(r): return r.choice(["matric", "inter"])
def dstyle(s): return s.style["date_style"]
def D(s, f): return parse_date(s.text[f])


# ---------------------------------------------------------------- field level
def missing_text(s, r):
    f = r.choice(REQUIRED_TEXT); s.text[f] = ""; s.status[f] = "MISSING"
    s.issues.append(mk(T.MISSING_REQUIRED, [f], f"Required field '{f}' is empty.")); return {f}

def missing_signature(s, r):
    s.signature = "none"; s.status["signature"] = "MISSING"
    s.issues.append(mk(T.MISSING_REQUIRED, ["signature"], "The signature box is empty.")); return {"signature"}

def signature_speck(s, r):  # only a stray dot or tiny mark: detector must still say MISSING
    s.signature = "speck"; s.status["signature"] = "MISSING"
    s.issues.append(mk(T.MISSING_REQUIRED, ["signature"], "The signature box has only a stray mark, no signature.")); return {"signature"}

def no_gender_or_program(s, r):
    g = r.choice(["gender", "program"]); ids = [k for k in s.ticks if k.startswith(g)]
    for k in ids: s.ticks[k] = False
    s.issues.append(mk(T.NO_SELECTION, ids, f"No box is ticked for {g}.")); return set(ids)

def unreadable(s, r):
    f = r.choice(REQUIRED_TEXT); s.defects[f] = "illegible"; s.status[f] = "UNREADABLE"
    s.issues.append(mk(T.UNREADABLE_FIELD, [f], f"The handwriting in '{f}' cannot be read.")); return {f}

def placeholder(s, r):
    f = r.choice(["father_name", "address"]); s.text[f] = r.choice(["N/A", "n/a", "-", "nil", "XXXX", "None", "......"])
    s.status[f] = "INVALID"; s.issues.append(mk(T.PLACEHOLDER_VALUE, [f], f"'{f}' contains a placeholder instead of a real value.")); return {f}

def correction(s, r):
    f = r.choice(["full_name", "father_name", "cnic", "date_of_birth", "email", "phone", "address", "declaration_name"])
    good = s.text[f]; chars = list(good); i = r.choice([k for k, c in enumerate(chars) if c.isalnum()])
    chars[i] = r.choice("0123456789") if chars[i].isdigit() else r.choice("abcdefghmnoprstu")
    s.defects[f] = ("strike", "".join(chars))
    s.issues.append(mk(T.CORRECTION_MARK, [f], f"'{f}' has a struck out value that was rewritten; please confirm the final value.")); return {f}

# ---------------------------------------------------------------- format
def cnic_bad(s, r):
    c = s.text["cnic"]; kind = r.choice(["truncated", "truncated", "extra", "nodash", "spaces", "confusable"])
    ch = list(c); dig = [i for i, x in enumerate(ch) if x.isdigit()]
    if kind == "truncated":
        for i in sorted(r.sample(dig, r.randint(1, 3)), reverse=True): del ch[i]
        out = "".join(ch)
    elif kind == "extra":
        ch.insert(r.choice(dig), str(r.randint(0, 9))); out = "".join(ch)
    elif kind == "nodash": out = c.replace("-", "")
    elif kind == "spaces": out = c.replace("-", " ")
    else:
        for a, b in (("0", "O"), ("1", "l"), ("5", "S"), ("8", "B")):
            if a in c: out = c.replace(a, b, 1); break
        else: out = c[:-2]
    if CNIC_RE.match(out): return None
    s.text["cnic"] = out; s.status["cnic"] = "INVALID"
    s.issues.append(mk(T.INVALID_CNIC, ["cnic"], f"CNIC '{out}' does not match #####-#######-# ({kind}).")); return {"cnic"}

def email_bad(s, r):
    e = s.text["email"]; local, dom = e.split("@")
    out = r.choice([f"{local}{dom}", f"{local}@{dom.split('.')[0]}", f"{local} @{dom}", f"{local}@@{dom}", f"@{dom}", f"{local}@{dom}.", f"{local}@{dom.replace('.', '..')}", f"{local},{dom}"])
    if EMAIL_RE.match(out): return None
    s.text["email"] = out; s.status["email"] = "INVALID"
    s.issues.append(mk(T.INVALID_EMAIL, ["email"], f"Email '{out}' is not a valid address.")); return {"email"}

def phone_bad(s, r):
    p = s.text["phone"]; d = p.replace("-", "")
    out = r.choice([f"04{d[2:4]}-{d[4:]}", p[:-2], p + "5", d, "042-35761234", f"{p[:4]} {p[5:]}", s.text["email"] if r.random() < .3 else p[:-3]])
    if PHONE_RE.match(out): return None
    s.text["phone"] = out; s.status["phone"] = "INVALID"
    s.issues.append(mk(T.INVALID_PHONE, ["phone"], f"Phone '{out}' does not match 03XX-XXXXXXX.")); return {"phone"}

def date_bad(s, r):
    f = r.choice(["date_of_birth", "declaration_date"]); y = r.randint(1998, 2008)
    out = r.choice([f"31/02/{y}", f"30/02/{y}", f"31/04/{y}", f"29/02/{y + 1 if y % 4 == 0 else y}",
                    f"45/03/{y}", f"12/13/{y}", "00/00/0000", f"15/00/{y}"])
    if parse_date(out): return None
    s.text[f] = out; s.status[f] = "INVALID"
    s.issues.append(mk(T.INVALID_DATE, [f], f"'{out}' in {f} is not a real calendar date.")); return {f}

def number_bad(s, r):
    f = r.choice(NUMERIC); k = f.split("_")[0]
    out = r.choice(["abc", "7OO", "seven hundred", "9x0", "20l8" if f.endswith("year") else "N/A?", "1,0S0"])
    s.text[f] = out; s.status[f] = "INVALID"
    s.issues.append(mk(T.INVALID_NUMBER, [f], f"'{out}' in {f} is not a valid number."))
    return {f, f"{k}_year", f"{k}_total", f"{k}_obtained", f"{k}_percentage"} if not f.endswith("year") else {f, "matric_year", "inter_year"}

# ---------------------------------------------------------------- cross field
def underage(s, r):
    d = add_years(s.today, -r.randint(8, 15)) - dt.timedelta(days=r.randint(1, 340))
    s.text["date_of_birth"] = fmt_date(d, dstyle(s))
    s.issues.append(mk(T.UNDERAGE, ["date_of_birth"], "Applicant is younger than 16.")); return {"date_of_birth"}

def decl_future(s, r):
    s.text["declaration_date"] = fmt_date(s.today + dt.timedelta(days=r.randint(1, 400)), dstyle(s))
    s.issues.append(mk(T.DECLARATION_DATE_FUTURE, ["declaration_date"], "Declaration date is in the future.")); return {"declaration_date"}

def decl_before_dob(s, r):
    d = D(s, "date_of_birth")
    if not d: return None
    s.text["declaration_date"] = fmt_date(d - dt.timedelta(days=r.randint(1, 1500)), dstyle(s))
    s.issues.append(mk(T.DECLARATION_BEFORE_DOB, ["date_of_birth", "declaration_date"], "Declaration date is before the date of birth.")); return {"date_of_birth", "declaration_date"}

def cnic_gender(s, r):
    if r.random() < .5:
        s.ticks["gender_male"], s.ticks["gender_female"] = s.ticks["gender_female"], s.ticks["gender_male"]
    else:
        c = s.text["cnic"]; s.text["cnic"] = c[:-1] + str((int(c[-1]) + r.choice([1, 3, 5, 7, 9])) % 10)
    s.issues.append(mk(T.CNIC_GENDER_MISMATCH, ["cnic", "gender_male", "gender_female"], "Last CNIC digit does not match the ticked gender.")); return {"cnic", "gender_male", "gender_female"}

def multiple(s, r):
    g = r.choice(["gender", "program"]); ids = [k for k in s.ticks if k.startswith(g)]
    cur = [k for k in ids if s.ticks[k]]; others = [k for k in ids if not s.ticks[k]]
    for k in r.sample(others, r.randint(1, min(2, len(others)))): s.ticks[k] = True
    s.issues.append(mk(T.MULTIPLE_SELECTIONS, ids, f"More than one box is ticked for {g}.")); return set(ids)

def year_order(s, r):
    m, i = int(s.text["matric_year"]), int(s.text["inter_year"])
    a, b = r.choice([(i, i), (i, m), (i + r.randint(1, 3), i)])
    s.text["matric_year"], s.text["inter_year"] = str(a), str(b)
    s.issues.append(mk(T.YEAR_ORDER, ["matric_year", "inter_year"], "Matric year must be earlier than intermediate year.")); return {"matric_year", "inter_year"}

def marks_exceed(s, r):
    k = exam(r); t = int(s.text[f"{k}_total"]); o = t + r.randint(1, 150)
    s.text[f"{k}_obtained"] = str(o); s.text[f"{k}_percentage"] = f"{o / t * 100:.2f}"
    s.issues.append(mk(T.MARKS_EXCEED_TOTAL, [f"{k}_total", f"{k}_obtained"], "Obtained marks exceed total marks.")); return {f"{k}_total", f"{k}_obtained", f"{k}_percentage"}

def pct_mismatch(s, r):
    k = exam(r); true = int(s.text[f"{k}_obtained"]) / int(s.text[f"{k}_total"]) * 100
    d = r.uniform(3, 20) * r.choice([-1, 1]); new = true + d
    if not 1 <= new <= 100: new = true - d
    s.text[f"{k}_percentage"] = f"{new:.2f}"
    s.issues.append(mk(T.PERCENTAGE_MISMATCH, [f"{k}_total", f"{k}_obtained", f"{k}_percentage"], "Stated percentage differs from obtained / total.")); return {f"{k}_total", f"{k}_obtained", f"{k}_percentage"}

def name_mismatch(s, r):
    a = s.text["full_name"]
    for _ in range(30):
        g = r.choice(["male", "female"]); b = pick_name(r, g, r.choice(SURNAMES)) if r.random() < .6 else f"{a.split()[0]} {r.choice(SURNAMES)}"
        if not names_match(a, b): break
    else: return None
    s.text["declaration_name"] = b
    s.issues.append(mk(T.NAME_MISMATCH, ["full_name", "declaration_name"], "Declaration name does not match the applicant name.")); return {"full_name", "declaration_name"}

ERRORS: Dict[str, Callable] = dict(
    missing_text=missing_text, missing_signature=missing_signature, signature_speck=signature_speck,
    no_gender_or_program=no_gender_or_program, unreadable=unreadable, placeholder=placeholder, correction=correction,
    cnic_bad=cnic_bad, email_bad=email_bad, phone_bad=phone_bad, date_bad=date_bad, number_bad=number_bad,
    underage=underage, decl_future=decl_future, decl_before_dob=decl_before_dob, cnic_gender=cnic_gender,
    multiple=multiple, year_order=year_order, marks_exceed=marks_exceed, pct_mismatch=pct_mismatch, name_mismatch=name_mismatch)


# ---------------------------------------------------------------- benign (must NOT raise issues)
def b_name_variant(s, r):
    a = s.text["full_name"]; parts = a.split()
    v = r.choice([a.upper(), a.lower(), " ".join(parts[:1] + parts[-1:]) if len(parts) > 2 else a.upper(),
                  a.replace("a", "e", 1) if "a" in a[1:] else a.lower(), a + ".", parts[-1] + " " + " ".join(parts[:-1])])
    if not names_match(a, v) or v == a: return None
    s.text["declaration_name"] = v; return {"declaration_name"}

def b_mixed_date_style(s, r):
    other = r.choice([x for x in DATE_STYLES if x != dstyle(s)])
    d = D(s, "declaration_date")
    if not d: return None
    s.text["declaration_date"] = fmt_date(d, other); return {"declaration_date"}

def b_age_exactly_16(s, r):
    s.text["date_of_birth"] = fmt_date(add_years(s.today, -16), dstyle(s)); return {"date_of_birth"}

def b_decl_today(s, r):
    s.text["declaration_date"] = fmt_date(s.today, dstyle(s)); return {"declaration_date"}

def b_full_marks(s, r):
    k = exam(r); s.text[f"{k}_obtained"] = s.text[f"{k}_total"]; s.text[f"{k}_percentage"] = "100.00"
    return {f"{k}_total", f"{k}_obtained", f"{k}_percentage"}

def b_percentage_symbol(s, r):
    k = exam(r); s.text[f"{k}_percentage"] = f"{int(s.text[f'{k}_obtained']) / int(s.text[f'{k}_total']) * 100:.1f}%"
    return {f"{k}_total", f"{k}_obtained", f"{k}_percentage"}

def b_faint_signature(s, r):
    s.signature = "faint"; return {"signature"}

BENIGN: Dict[str, Callable] = dict(name_variant=b_name_variant, mixed_date_style=b_mixed_date_style, age_exactly_16=b_age_exactly_16,
                                   decl_today=b_decl_today, full_marks=b_full_marks, pct_symbol=b_percentage_symbol, faint_signature=b_faint_signature)


def try_apply(s: FormState, name: str, r: random.Random, touched: set, table) -> bool:
    """Apply injector on a copy; commit only if it applies and touches no already used field."""
    c = copy.deepcopy(s)
    try: out = table[name](c, r)
    except (ValueError, TypeError, ZeroDivisionError, IndexError): return False  # field already corrupted by another injector
    if not out or (out & touched): return False
    s.__dict__.update(c.__dict__); touched |= out
    (s.applied if table is ERRORS else s.benign).append(name); return True
