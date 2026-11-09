"""A stand-in for FieldOCR that "reads" from ground truth with controlled noise.

Used by ``scripts/evaluate_all.py --dry-run`` and by tests, to exercise the whole evaluation chain
(alignment, ink detection, rules, scoring, error analysis) without OCR models. The numbers it
produces describe the noise setting, NOT the real system.

The pipeline asks for text field by field, in template order, only for boxes that contain ink,
so the stand-in just hands out the true texts of the non-empty text fields in that order.
"""
from __future__ import annotations

import random
from typing import Sequence

from formsense.vision.ocr import OCRResult


class SimulatedOCR:
    def __init__(self, texts: Sequence[str], noise: float = 0.12, seed: int = 0):
        self.queue = list(texts)
        self.noise = noise
        self.rng = random.Random(seed)
        self.calls = 0

    def read(self, crop) -> OCRResult:
        self.calls += 1
        text = self.queue.pop(0) if self.queue else ""
        r = self.rng.random()
        if text and r < self.noise:                                   # a misread character, low confidence
            i = self.rng.randrange(len(text))
            wrong = self.rng.choice("0123456789abcdefghijklmnopqrstuvwxyz")
            return OCRResult(text[:i] + wrong + text[i + 1:], 0.45, "simulated")
        if text and r < self.noise * 1.4:                              # correct text but flagged unreadable
            return OCRResult(text, 0.30, "simulated")
        return OCRResult(text, 0.95, "simulated")
