"""Value extraction: from per field OCR output to finished ``FieldResult`` objects.

Input is what the vision side produces for each field crop (``raw_text``,
``ocr_confidence``, ``ink_ratio``, optionally ``checked``). Output is the same
``FieldResult`` with ``status`` and ``value`` filled in, plus human readable notes.

Status logic (from the project brief)
-------------------------------------
* checkbox   : ``checked`` is trusted if the detector set it, else derived from ink.
* signature  : enough ink = OK, otherwise MISSING (a stray dot is not a signature).
* text/date  : no ink and no text = MISSING; ink but nothing read, or OCR confidence
               below the threshold = UNREADABLE; text that fails normalisation or is a
               placeholder ("N/A", "-") = INVALID; otherwise OK.

Defensive choices for real data
-------------------------------
* If OCR returned text but ``ink_ratio`` says the box is empty, the text wins and a note
  is added (the ink measurement may be unavailable; a default of 0.0 must not wipe out a
  successful read).
* OCR confusion repair (O/0, l/1 ...) runs only when OCR confidence is below
  ``repair_below_conf``. A confident read of the letter O in a CNIC is the applicant's
  mistake, not an OCR slip, so it must stay visible to the rules.
* ``raw_text`` is never modified: it stays exactly what OCR returned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

from formsense.nlp.normalize import (
    DEFAULT_FORMATS,
    STRICT_VALUE_FORMATS,
    clean_text,
    is_placeholder,
    normalize_value,
    pick_format,
)
from formsense.nlp.template_info import FieldInfo, TemplateInfo
from formsense.schema import FieldResult, FieldStatus
from formsense.thresholds import OCR_UNREADABLE_BELOW

log = logging.getLogger("formsense.nlp")


def _upd(fr: FieldResult, changes: Dict) -> FieldResult:
    """Copy a FieldResult with changes; works on pydantic v1 and v2."""
    return fr.model_copy(update=changes) if hasattr(fr, "model_copy") else fr.copy(update=changes)


@dataclass(frozen=True)
class ExtractionConfig:
    ocr_conf_min: float = OCR_UNREADABLE_BELOW   # below this, ink + text is UNREADABLE (formsense/thresholds.py)
    repair_below_conf: float = 0.90     # attempt OCR confusion repair only below this
    ink_empty_below: float = 0.00682      # text/date boxes: less ink than this = empty
    signature_ink_min: float = 0.00692     # signature boxes: at least this much ink = present
    checkbox_ink_min: float = 0.08997      # checkboxes without a detector value: ink = ticked
    use_dateparser: bool = True
    formats: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FORMATS))
    # NOTE: the ink defaults are starting points. Run `python -m formsense.nlp.calibrate`
    # on your labels and copy the suggested numbers here (or load them from a file).


@dataclass
class Extraction:
    result: FieldResult
    notes: List[str] = field(default_factory=list)


def extract_field(fr: FieldResult, info: FieldInfo, cfg: ExtractionConfig = ExtractionConfig()) -> Extraction:
    notes: List[str] = []
    ink = fr.ink_ratio or 0.0

    # ---- checkbox ----
    if info.type == "checkbox":
        checked = fr.checked if fr.checked is not None else ink >= cfg.checkbox_ink_min
        return Extraction(_upd(fr, {"checked": checked, "status": FieldStatus.OK, "value": None}), notes)

    # ---- signature ----
    if info.type == "signature":
        present = ink >= cfg.signature_ink_min
        status = FieldStatus.OK if present else FieldStatus.MISSING
        if not present and ink > 0:
            notes.append("signature box has ink below the signature threshold (stray mark?)")
        return Extraction(_upd(fr, {"status": status, "value": None}), notes)

    # ---- text / date ----
    text = clean_text(fr.raw_text)
    conf = fr.ocr_confidence
    no_ink = ink < cfg.ink_empty_below

    if no_ink and not text:
        return Extraction(_upd(fr, {"status": FieldStatus.MISSING, "value": None}), notes)
    if no_ink and text:
        notes.append("OCR text present but ink_ratio is near zero; trusting the text")

    if not text:
        notes.append("ink present but OCR returned nothing")
        return Extraction(_upd(fr, {"status": FieldStatus.UNREADABLE, "value": None}), notes)

    if conf is not None and conf < cfg.ocr_conf_min:
        notes.append(f"OCR confidence {conf:.2f} below {cfg.ocr_conf_min:.2f}")
        return Extraction(_upd(fr, {"status": FieldStatus.UNREADABLE, "value": None}), notes)

    fmt = pick_format(info.name, info.type, info.kind, cfg.formats)

    if is_placeholder(text):
        value = None if fmt in STRICT_VALUE_FORMATS else text
        notes.append("placeholder text")
        return Extraction(_upd(fr, {"status": FieldStatus.INVALID, "value": value}), notes)

    repair = conf is not None and conf < cfg.repair_below_conf
    kwargs = {"use_dateparser": cfg.use_dateparser} if fmt == "date" else {}
    norm = normalize_value(fmt, text, repair=repair, **kwargs)

    if norm.ok:
        if norm.repaired:
            notes.append("value repaired from likely OCR confusions")
        return Extraction(_upd(fr, {"status": FieldStatus.OK, "value": norm.value}), notes)

    notes.append(f"normalisation failed: {norm.reason}")
    value = None if fmt in STRICT_VALUE_FORMATS else text
    return Extraction(_upd(fr, {"status": FieldStatus.INVALID, "value": value}), notes)


def extract_fields(
    fields: Sequence[FieldResult],
    template: Union[TemplateInfo, Mapping, str, None] = None,
    cfg: ExtractionConfig = ExtractionConfig(),
) -> List[Extraction]:
    tpl = TemplateInfo.coerce(template)
    out: List[Extraction] = []
    for fr in fields:
        info = tpl.fields.get(fr.name)
        if info is None:
            log.warning("field %r is not in the template; passed through unchanged", fr.name)
            out.append(Extraction(fr, ["field not in template"]))
            continue
        try:
            out.append(extract_field(fr, info, cfg))
        except Exception:  # one bad field must not sink the whole form
            log.exception("extraction failed for field %r; marking UNREADABLE", fr.name)
            out.append(Extraction(_upd(fr, {"status": FieldStatus.UNREADABLE}), ["extraction error"]))
    return out


def enrich_fields(
    fields: Sequence[FieldResult],
    template: Union[TemplateInfo, Mapping, str, None] = None,
    cfg: ExtractionConfig = ExtractionConfig(),
) -> List[FieldResult]:
    """Convenience wrapper that returns only the finished FieldResults."""
    return [e.result for e in extract_fields(fields, template, cfg)]
