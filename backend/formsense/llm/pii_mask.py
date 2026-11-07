"""Mask personal data before it reaches an LLM.

The brief is explicit: CNIC, phone and email must be masked before the LLM call. This
module is the one place that decides what "masked" means, so every caller gets the same
behaviour and nothing forgets to mask a field.

Design
------
* Masking works on ``FieldResult`` objects (after extraction) or on raw strings.
* Three field names are ALWAYS masked, regardless of configuration: ``cnic``, ``phone``,
  ``email``. A caller cannot turn this off; it can only add more fields to mask.
* Masking keeps enough shape for the LLM to still reason about format problems (an
  invalid CNIC still looks the wrong shape after masking) without revealing the value:
    CNIC    35202-1234567-1   -> 35XXX-XXXXXXX-X
    phone   0300-1234567      -> 03XX-XXXXXXX
    email   ali.raza@x.com    -> a***@x.com
* A field already judged MISSING or UNREADABLE has no text to leak, so it is left as is.
* The mapping from masked placeholder back to the real value is returned separately and
  is never part of what gets sent anywhere; it exists only so a caller can, if it chooses,
  substitute real values back into a report meant for the applicant themselves (not for
  the LLM, not for a log).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Sequence, Union

from formsense.schema import FieldResult

ALWAYS_MASKED: FrozenSet[str] = frozenset({"cnic", "phone", "email"})

_EMAIL_LOCAL = re.compile(r"^(.)(.*)(@.*)$")


@dataclass(frozen=True)
class MaskResult:
    fields: List[FieldResult]                  # same fields, PII values replaced
    unmasked_by_name: Dict[str, str] = field(default_factory=dict)   # name -> real value, for local use only


def mask_cnic(value: str) -> str:
    if not re.match(r"^\d{5}-\d{7}-\d$", value):
        return "[REDACTED CNIC]"
    return f"{value[:2]}XXX-XXXXXXX-X"


def mask_phone(value: str) -> str:
    if not re.match(r"^03\d{2}-\d{7}$", value):
        return "[REDACTED PHONE]"
    return f"{value[:2]}XX-XXXXXXX"


def mask_email(value: str) -> str:
    m = _EMAIL_LOCAL.match(value)
    if not m or "@" not in value:
        return "[REDACTED EMAIL]"
    first, _, domain = m.groups()
    return f"{first}***{domain}"


_MASKERS = {"cnic": mask_cnic, "phone": mask_phone, "email": mask_email}


def mask_generic(value: str) -> str:
    """Fallback for a caller-specified extra field with no dedicated masker.

    Keeps the first two characters (useful for sanity-checking in a demo) and replaces
    the rest with a placeholder of fixed length, so the output never reveals the real
    length of the value either.
    """
    head = value[:2] if len(value) > 2 else ""
    return f"{head}***" if head else "[REDACTED]"


def mask_value(field_name: str, value: str, known_only: bool = True) -> str:
    """Mask one value by the field name it came from.

    Field names with a dedicated masker (cnic, phone, email) use it. Any other field
    name passes through unchanged by default (``known_only=True``). Pass
    ``known_only=False`` to mask an unrecognised field generically instead; that is what
    ``mask_fields`` does for a caller-supplied ``extra_fields`` name, since the caller
    has explicitly said that field must not reach the LLM.
    """
    masker = _MASKERS.get(field_name)
    if masker:
        return masker(value)
    return value if known_only else mask_generic(value)


def mask_fields(
    fields: Sequence[FieldResult],
    extra_fields: Sequence[str] = (),
) -> MaskResult:
    """Mask CNIC, phone and email (always) plus any names in ``extra_fields``.

    Only ``raw_text`` and ``value`` are replaced. Status, ink_ratio, confidence and bbox
    are left alone, since an LLM explaining "the signature is missing" needs the status,
    not the text.
    """
    to_mask = ALWAYS_MASKED | set(extra_fields)
    out: List[FieldResult] = []
    unmasked: Dict[str, str] = {}
    for f in fields:
        if f.name not in to_mask:
            out.append(f)
            continue
        real = f.value or f.raw_text
        if not real:                      # MISSING or UNREADABLE: nothing to mask
            out.append(f)
            continue
        unmasked[f.name] = real
        masked = mask_value(f.name, real, known_only=False)
        changes = {"raw_text": masked, "value": masked if f.value else f.value}
        out.append(f.model_copy(update=changes) if hasattr(f, "model_copy") else f.copy(update=changes))
    return MaskResult(fields=out, unmasked_by_name=unmasked)


def mask_text(text: str, fields: Sequence[FieldResult], extra_fields: Sequence[str] = ()) -> str:
    """Replace any occurrence of a masked field's real value inside free text.

    Use this on anything built from the original fields before sending it onward, for
    example a free-text note that happens to quote the applicant's phone number.
    """
    to_mask = ALWAYS_MASKED | set(extra_fields)
    out = text
    for f in fields:
        if f.name not in to_mask:
            continue
        real = f.value or f.raw_text
        if real and real in out:
            out = out.replace(real, mask_value(f.name, real, known_only=False))
    return out


def assert_no_pii(text: str, fields: Sequence[FieldResult], extra_fields: Sequence[str] = ()) -> None:
    """Raise if any real CNIC, phone or email value is still present in ``text``.

    A cheap safety net to call right before an LLM request or a log line, independent of
    whether masking was actually applied upstream.
    """
    to_mask = ALWAYS_MASKED | set(extra_fields)
    for f in fields:
        if f.name not in to_mask:
            continue
        real = f.value or f.raw_text
        if real and len(real) >= 5 and real in text:
            raise ValueError(f"unmasked value for field '{f.name}' is present in text sent onward")
