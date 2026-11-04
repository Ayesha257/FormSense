"""OCR for one field crop: PaddleOCR first, TrOCR (handwriting) as fallback.

Models load lazily on first use, so importing this module is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FALLBACK_BELOW = 0.80   # run TrOCR when PaddleOCR confidence is below this
TROCR_MODEL = "microsoft/trocr-base-handwritten"


@dataclass
class OCRResult:
    text: str
    confidence: float     # 0..1
    engine: str           # "paddle", "trocr" or "none"


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


class PaddleReader:
    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        try:  # PaddleOCR 3.x
            # enable_mkldnn=False: PaddlePaddle 3.3 crashes on CPU with oneDNN on
            self._ocr = PaddleOCR(lang=lang, use_doc_orientation_classify=False,
                                  use_doc_unwarping=False, use_textline_orientation=False,
                                  enable_mkldnn=False)
            self._v3 = True
        except (TypeError, ValueError):  # PaddleOCR 2.x
            self._ocr = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)
            self._v3 = False

    def lines(self, img: np.ndarray) -> list[tuple[str, float, tuple[float, float]]]:
        """[(text, score, (x, y) of line's top-left)] for every detected line."""
        img = _to_bgr(img)
        out = []
        if self._v3:
            for res in self._ocr.predict(img):
                polys = res.get("rec_polys") or res.get("dt_polys") or []
                for i, (t, s) in enumerate(zip(res["rec_texts"], res["rec_scores"])):
                    if i < len(polys):
                        p = np.asarray(polys[i])
                        xy = (float(p[:, 0].min()), float(p[:, 1].min()))
                    else:
                        xy = (0.0, float(i))
                    out.append((t, float(s), xy))
        else:
            for box, (t, s) in (self._ocr.ocr(img)[0] or []):
                p = np.asarray(box)
                out.append((t, float(s), (float(p[:, 0].min()), float(p[:, 1].min()))))
        return out

    def read(self, img: np.ndarray) -> OCRResult:
        lines = [l for l in self.lines(img) if l[0].strip()]
        if not lines:
            return OCRResult("", 0.0, "paddle")
        # reading order: top-to-bottom (rows ~ 12 px), then left-to-right
        lines.sort(key=lambda l: (round(l[2][1] / 12), l[2][0]))
        text = " ".join(l[0].strip() for l in lines)
        weights = [max(1, len(l[0])) for l in lines]
        conf = float(np.average([l[1] for l in lines], weights=weights))
        return OCRResult(text, conf, "paddle")


class TrOCRReader:
    def __init__(self, model_name: str = TROCR_MODEL):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self._torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()

    def read(self, img: np.ndarray) -> OCRResult:
        from PIL import Image
        rgb = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        with self._torch.no_grad():
            out = self.model.generate(pixel_values, max_new_tokens=64,
                                      output_scores=True, return_dict_in_generate=True)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        scores = self.model.compute_transition_scores(out.sequences, out.scores,
                                                      normalize_logits=True)
        conf = float(np.exp(scores[0].numpy().mean())) if scores.numel() else 0.0
        return OCRResult(text, conf, "trocr")


class FieldOCR:
    """PaddleOCR, falling back to TrOCR when Paddle is unsure."""

    def __init__(self, use_trocr: bool = True, fallback_below: float = FALLBACK_BELOW):
        self.use_trocr = use_trocr
        self.fallback_below = fallback_below
        self._paddle: PaddleReader | None = None
        self._trocr: TrOCRReader | None = None

    @property
    def paddle(self) -> PaddleReader:
        if self._paddle is None:
            self._paddle = PaddleReader()
        return self._paddle

    @property
    def trocr(self) -> TrOCRReader:
        if self._trocr is None:
            self._trocr = TrOCRReader()
        return self._trocr

    def read(self, crop: np.ndarray) -> OCRResult:
        first = self.paddle.read(crop)
        if not self.use_trocr or first.confidence >= self.fallback_below:
            return first
        second = self.trocr.read(crop)
        return second if second.confidence > first.confidence else first
