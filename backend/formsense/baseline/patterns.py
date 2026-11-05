"""Regex patterns the baseline uses to pull values out of plain OCR text.

Kept separate from the main pipeline's extraction on purpose: the baseline
must stay simple so the comparison is honest.
"""
from __future__ import annotations

import re

from formsense.schema import FieldSpec, FieldType

CNIC = re.compile(r"\b\d{5}[-\s]?\d{7}[-\s]?\d\b")
DATE = re.compile(r"\b\d{1,2}\s?[/\-.]\s?\d{1,2}\s?[/\-.]\s?\d{2,4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?:\+92|0)\s?3\d{2}[-\s]?\d{7}\b")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

# characters a text-only system might read for a tick mark next to an option
TICK = re.compile(r"(?:^|\s)[✓✔☑√xXvV✗]\s*$")


def pattern_for(spec: FieldSpec) -> re.Pattern | None:
    """Which regex fits this field, guessed from its name and type."""
    n = spec.name.lower()
    if "cnic" in n or "b_form" in n:
        return CNIC
    if spec.type == FieldType.DATE or "date" in n:
        return DATE
    if "email" in n:
        return EMAIL
    if "phone" in n or "mobile" in n:
        return PHONE
    if n.endswith("_year"):
        return YEAR
    if any(k in n for k in ("marks", "obtained", "total", "percentage")):
        return NUMBER
    return None


def label_regex(label: str) -> re.Pattern | None:
    """'Father's Name' -> matches 'FATHERS NAME', "father's name:" etc."""
    tokens = re.findall(r"[a-z0-9]+", label.lower())
    if not tokens:
        return None
    return re.compile(r"[^a-z0-9]*".join(map(re.escape, tokens)), re.IGNORECASE)
