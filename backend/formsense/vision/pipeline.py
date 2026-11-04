"""Vision step: filled page image -> list[FieldResult].

    page -> preprocess -> align to template -> new-ink mask
         -> per field: ink ratio -> checkbox / signature / empty decision
         -> OCR on text and date fields that have ink
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formsense.ingest.loader import preprocess
from formsense.schema import FieldResult, FieldStatus, FieldType, FormTemplate
from formsense.vision import fields as F
from formsense.vision.align import AlignResult, align_to_template
from formsense.vision.ocr import FieldOCR

UNREADABLE_BELOW = 0.60   # OCR confidence below this = UNREADABLE


@dataclass
class VisionOutput:
    fields: list[FieldResult]
    aligned: np.ndarray       # page in template coordinates (grayscale)
    new_ink: np.ndarray       # applicant's ink mask
    alignment: AlignResult


def analyze_page(page: np.ndarray, template: FormTemplate, blank: np.ndarray,
                 ocr: FieldOCR | None = None, clean: bool = True) -> VisionOutput:
    """Analyse one page. Pass ocr=None to skip OCR (fast; detectors only)."""
    gray = preprocess(page) if clean else page
    al = align_to_template(gray, blank)
    new_ink = F.new_ink_mask(al.image, blank)

    results: list[FieldResult] = []
    for spec in template.fields:
        ratio = F.ink_ratio(new_ink, spec.bbox)
        r = FieldResult(name=spec.name, bbox=spec.bbox, ink_ratio=round(ratio, 4))

        if spec.type == FieldType.CHECKBOX:
            r.checked = ratio >= F.CHECKBOX_MIN
            r.value = "checked" if r.checked else "unchecked"

        elif spec.type == FieldType.SIGNATURE:
            signed = ratio >= F.SIGNATURE_MIN
            r.value = "signed" if signed else None
            r.status = FieldStatus.OK if signed else FieldStatus.MISSING

        else:  # TEXT or DATE
            if ratio < F.EMPTY_TEXT_MAX:
                r.status = FieldStatus.MISSING
            elif ocr is not None:
                res = ocr.read(F.clean_crop_for_ocr(al.image, blank, spec.bbox, new_ink))
                r.raw_text = res.text
                r.ocr_confidence = round(res.confidence, 3)
                r.status = (FieldStatus.UNREADABLE
                            if not res.text or res.confidence < UNREADABLE_BELOW
                            else FieldStatus.OK)
        results.append(r)

    return VisionOutput(results, al.image, new_ink, al)
