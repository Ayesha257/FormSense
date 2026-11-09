"""Vision step: filled page image -> list[FieldResult].

    page -> preprocess -> align to template -> new-ink mask
         -> per field: ink ratio -> checkbox / signature / empty decision
         -> OCR on text and date fields that have ink
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from formsense.ingest.loader import preprocess
from formsense.schema import FieldResult, FieldStatus, FieldType, FormTemplate
from formsense.vision import fields as F
from formsense.vision.align import AlignResult, align_to_template
from formsense.vision.ocr import FieldOCR

from formsense.thresholds import OCR_UNREADABLE_BELOW as UNREADABLE_BELOW  # noqa: E402  (one shared value)


@dataclass
class VisionOutput:
    fields: list[FieldResult]
    aligned: np.ndarray       # page in template coordinates (grayscale)
    new_ink: np.ndarray       # applicant's ink mask
    alignment: AlignResult
    shifts: dict = field(default_factory=dict)   # per-field (dx, dy) from local refinement


def analyze_page(page: np.ndarray, template: FormTemplate, blank: np.ndarray,
                 ocr: FieldOCR | None = None, clean: bool = True) -> VisionOutput:
    """Analyse one page. Pass ocr=None to skip OCR (fast; detectors only)."""
    gray = preprocess(page) if clean else page
    al = align_to_template(gray, blank)
    page_dark = F.darkness(al.image)
    blank_dark = F.darkness(blank)
    new_ink = np.zeros_like(al.image)        # applicant's ink, for display and debugging
    shifts: dict[str, tuple[int, int]] = {}

    results: list[FieldResult] = []
    for spec in template.fields:
        dx, dy = F.local_shift(page_dark, blank_dark, spec.bbox) if al.ok else (0, 0)
        shifts[spec.name] = (dx, dy)
        x, y, w, h = spec.bbox
        ink = F.field_ink(page_dark, blank_dark, spec.bbox, (dx, dy))
        ratio = F.mask_ratio(ink)
        r = FieldResult(name=spec.name, bbox=(x + dx, y + dy, w, h), ink_ratio=round(ratio, 4))

        if spec.type == FieldType.CHECKBOX:
            r.checked = ratio >= F.CHECKBOX_MIN
            r.value = "checked" if r.checked else "unchecked"

        elif spec.type == FieldType.SIGNATURE:
            # light pencil must still count, so signatures use the faint-ink threshold
            ink = F.field_ink(page_dark, blank_dark, spec.bbox, (dx, dy), min_darkness=F.FAINT_DARKNESS)
            ratio = F.mask_ratio(ink, inset=max(2, h // 12))
            r.ink_ratio = round(ratio, 4)
            signed = ratio >= F.SIGNATURE_MIN
            r.value = "signed" if signed else None
            r.status = FieldStatus.OK if signed else FieldStatus.MISSING

        else:  # TEXT or DATE
            if ratio < F.EMPTY_TEXT_MAX:
                r.status = FieldStatus.MISSING
            elif ocr is not None:
                res = ocr.read(F.clean_crop_for_ocr(al.image, blank, spec.bbox, ink, (dx, dy)))
                r.raw_text = res.text
                r.ocr_confidence = round(res.confidence, 3)
                r.status = (FieldStatus.UNREADABLE
                            if not res.text or res.confidence < UNREADABLE_BELOW
                            else FieldStatus.OK)
        F.paste(new_ink, ink, x + dx, y + dy)
        results.append(r)

    return VisionOutput(results, al.image, new_ink, al, shifts)
