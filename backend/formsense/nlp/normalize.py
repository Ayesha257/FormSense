"""Value normalisers: turn raw OCR text into clean, comparable values.

Design rules
------------
1. Strict by default. A CNIC that is missing a dash IS a form error, so the default
   path never "fixes" anything. Hiding a real applicant mistake would defeat the audit.
2. Repair only when OCR is unsure. When the caller passes ``repair=True`` (the extractor
   does this only for low confidence reads) common OCR confusions are fixed: O for 0,
   l for 1, S for 5, lost dashes, "+92" prefixes and so on. The result is flagged
   ``repaired=True`` so it can be reported.
3. Never raise. Every function returns a ``Norm``; bad input gives ``ok=False`` with a
   short machine readable reason.
4. No personal data in reasons or logs.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

from rapidfuzz import fuzz

# ---------- patterns (identical to the generator's reference validator) ----------
CNIC_RE = re.compile(r"^\d{5}-\d{7}-\d$")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^03\d{2}-\d{7}$")
YEAR_RE = re.compile(r"^\d{4}$")
PLACEHOLDER_RE = re.compile(r"^(n/?a|nil|none|null|x+|[-._\s]+)$", re.I)

DEFAULT_DATE_FORMATS: Sequence[str] = (
    "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y", "%d %B %Y", "%Y-%m-%d",
)

# Characters OCR engines commonly confuse with digits (used only in digit contexts).
_OCR_DIGIT = str.maketrans("OoQDIl|!SBZG", "000011115826")
_DASH_VARIANTS = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")


@dataclass(frozen=True)
class Norm:
    value: Optional[str]            # cleaned value, None when the text cannot be interpreted
    ok: bool
    reason: Optional[str] = None    # short code such as "cnic_format" when not ok
    repaired: bool = False          # True when OCR confusion repair changed the text


# ---------- helpers ----------
def clean_text(raw: Optional[str]) -> str:
    """Unicode normalise, unify dash characters, collapse whitespace."""
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).replace("\u200b", "")
    s = s.translate(_DASH_VARIANTS)
    return re.sub(r"\s+", " ", s).strip()


def _digitise(s: str) -> str:
    return s.translate(_OCR_DIGIT)


def is_placeholder(raw: Optional[str]) -> bool:
    return bool(PLACEHOLDER_RE.match(clean_text(raw)))


def _fail(reason: str) -> Norm:
    return Norm(None, False, reason)


# ---------- CNIC / phone / email ----------
def normalize_cnic(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw)
    if CNIC_RE.match(s):
        return Norm(s, True)
    if not repair:
        return _fail("cnic_format")
    digits = re.sub(r"\D", "", _digitise(s))
    if len(digits) == 13:
        return Norm(f"{digits[:5]}-{digits[5:12]}-{digits[12]}", True, repaired=True)
    return _fail("cnic_format")


def normalize_phone(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw)
    if PHONE_RE.match(s):
        return Norm(s, True)
    if not repair:
        return _fail("phone_format")
    digits = re.sub(r"\D", "", _digitise(s))
    if digits.startswith("0092"):
        digits = "0" + digits[4:]
    elif digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    if len(digits) == 11 and digits.startswith("03"):
        return Norm(f"{digits[:4]}-{digits[4:]}", True, repaired=True)
    return _fail("phone_format")


def normalize_email(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw)
    if EMAIL_RE.match(s):
        return Norm(s, True)
    if not repair:
        return _fail("email_format")
    t = re.sub(r"\s*[\(\[]\s*at\s*[\)\]]\s*", "@", s, flags=re.I)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r",(com|net|org|edu|pk)$", r".\1", t, flags=re.I)
    t = t.strip(".,;:")
    if EMAIL_RE.match(t):
        return Norm(t, True, repaired=True)
    return _fail("email_format")


# ---------- dates ----------
def parse_date_strict(text: Optional[str], formats: Sequence[str] = DEFAULT_DATE_FORMATS) -> Optional[dt.date]:
    t = clean_text(text)
    if not t:
        return None
    for f in formats:
        try:
            return dt.datetime.strptime(t, f).date()
        except ValueError:
            continue
    return None


def _dateparser_parse(s: str) -> Optional[dt.date]:
    """Fallback for human styles such as '5th March 2005'. Strict: day, month and year required."""
    try:
        import dateparser  # lazy: import is slow and the dependency is optional
    except ImportError:
        return None
    try:
        d = dateparser.parse(
            s,
            languages=["en"],
            settings={
                "DATE_ORDER": "DMY",
                "STRICT_PARSING": True,
                "REQUIRE_PARTS": ["day", "month", "year"],
            },
        )
    except Exception:  # dateparser can raise on odd input; never let that escape
        return None
    if d is None or not (1900 <= d.year <= 2100):
        return None
    return d.date()


def normalize_date(
    raw: Optional[str],
    repair: bool = False,
    formats: Sequence[str] = DEFAULT_DATE_FORMATS,
    use_dateparser: bool = True,
) -> Norm:
    s = clean_text(raw)
    d = parse_date_strict(s, formats)
    repaired = False
    if d is None and repair and re.fullmatch(r"[0-9OoQDIl|!SBZG/.\- ]+", s) and re.search(r"\d", s):
        d = parse_date_strict(_digitise(s), formats)
        repaired = d is not None
    if d is None and use_dateparser and s:
        d = _dateparser_parse(s)
    if d is None:
        return _fail("date_unparseable")
    return Norm(d.isoformat(), True, repaired=repaired)


# ---------- numbers and years ----------
_THOUSANDS_RE = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_PLAIN_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")


def normalize_number(raw: Optional[str], repair: bool = False) -> Norm:
    """Marks and percentages. Accepts a trailing %, and thousands commas ('1,050')."""
    s = clean_text(raw).rstrip("%").strip()
    repaired = False
    if repair and re.fullmatch(r"[0-9OoQDIl|!SBZG.,\s]+", s) and re.search(r"\d", s):
        t = re.sub(r"\s+", "", _digitise(s))
        repaired = t != s
        s = t
    if _THOUSANDS_RE.match(s):
        s = s.replace(",", "")
    if _PLAIN_NUMBER_RE.match(s):
        return Norm(s, True, repaired=repaired)
    return _fail("number_format")


def to_number(raw: Optional[str]) -> Optional[float]:
    n = normalize_number(raw)
    return float(n.value) if n.ok and n.value is not None else None


def normalize_year(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw)
    if YEAR_RE.match(s):
        return Norm(s, True)
    if repair and re.fullmatch(r"[0-9OoQDIl|!SBZG]{4}", s) and re.search(r"\d", s):
        t = _digitise(s)
        if YEAR_RE.match(t):
            return Norm(t, True, repaired=True)
    return _fail("year_format")


# ---------- names and free text ----------
def normalize_name(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw).strip(" ,.;:_|~'\"")
    if not re.search(r"[^\W\d_]", s):
        return _fail("name_no_letters")
    return Norm(s, True)


def normalize_text(raw: Optional[str], repair: bool = False) -> Norm:
    s = clean_text(raw)
    return Norm(s, True) if s else _fail("empty")


def _canon_name(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower())


def names_match(a: str, b: str, threshold: float = 80) -> bool:
    """Fuzzy equality for person names (order insensitive, case and punctuation insensitive)."""
    return fuzz.token_sort_ratio(_canon_name(a), _canon_name(b)) >= threshold


# ---------- dispatch ----------
Normalizer = Callable[..., Norm]

NORMALIZERS: Dict[str, Normalizer] = {
    "cnic": normalize_cnic,
    "phone": normalize_phone,
    "email": normalize_email,
    "date": normalize_date,
    "number": normalize_number,
    "year": normalize_year,
    "name": normalize_name,
    "text": normalize_text,
}

# Field names whose format is not implied by their type. Override per template if needed.
DEFAULT_FORMATS: Mapping[str, str] = {"cnic": "cnic", "email": "email", "phone": "phone"}

# Formats for which a failed normalisation must not keep the raw text as the value.
STRICT_VALUE_FORMATS = frozenset({"date", "number", "year"})


def pick_format(name: str, ftype: str, kind: str, overrides: Optional[Mapping[str, str]] = None) -> str:
    table = overrides if overrides is not None else DEFAULT_FORMATS
    if name in table:
        return table[name]
    if ftype == "date":
        return "date"
    if kind == "number":
        return "number"
    if name.endswith("_year"):
        return "year"
    if name.endswith("_name") or name == "name":
        return "name"
    return "text"


def normalize_value(fmt: str, raw: Optional[str], repair: bool = False, **kwargs) -> Norm:
    fn = NORMALIZERS.get(fmt, normalize_text)
    if fn is normalize_date:
        return fn(raw, repair=repair, **kwargs)
    return fn(raw, repair=repair)
