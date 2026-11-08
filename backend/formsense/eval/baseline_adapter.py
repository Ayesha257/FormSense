from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from formsense.baseline.ocr_regex import BaselineAuditor
from formsense.schema import FieldResult, FormTemplate
from formsense.vision.ocr import PaddleReader


class BaselineSystem:
    """Wraps the OCR+regex baseline so eval treats it like the full pipeline."""

    name = "baseline"

    def __init__(self, template: FormTemplate, reader: PaddleReader | None = None):
        self.auditor = BaselineAuditor(template, reader or PaddleReader())

    def extract(self, image: np.ndarray | str | Path) -> list[FieldResult]:
        if not isinstance(image, np.ndarray):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError("could not read form image")
        return self.auditor.run(image).fields