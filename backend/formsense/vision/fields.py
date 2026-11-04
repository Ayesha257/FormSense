"""Crop field boxes and measure the handwriting ("new ink") inside them.

Key idea: compare the aligned page with the blank template. Anything dark on
the page that is NOT already printed on the template is the applicant's ink.
That removes box borders, labels and guide lines without hand-tuning.
"""
from __future__ import annotations

import cv2
import numpy as np

from formsense.schema import BBox

# Tunable thresholds (share of the box area covered by new ink)
EMPTY_TEXT_MAX = 0.004     # text/date box below this = nothing written
CHECKBOX_MIN = 0.06        # checkbox above this = ticked
SIGNATURE_MIN = 0.008      # signature box above this = signed


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) robust to uneven lighting in phone photos."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def new_ink_mask(aligned: np.ndarray, blank: np.ndarray, tolerance_px: int = 3) -> np.ndarray:
    """Ink on the filled page that isn't printed on the blank template."""
    page_ink = ink_mask(aligned)
    printed = ink_mask(blank)
    # grow printed strokes a little so small alignment errors don't count as ink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    printed = cv2.dilate(printed, k)
    new = cv2.bitwise_and(page_ink, cv2.bitwise_not(printed))
    # drop isolated specks (scanner noise)
    return cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def crop(img: np.ndarray, bbox: BBox, inset: int = 0) -> np.ndarray:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, y0 = max(0, x + inset), max(0, y + inset)
    x1, y1 = min(W, x + w - inset), min(H, y + h - inset)
    return img[y0:y1, x0:x1]


def ink_ratio(new_ink: np.ndarray, bbox: BBox, inset: int = 2) -> float:
    c = crop(new_ink, bbox, inset)
    return float((c > 0).mean()) if c.size else 0.0


def clean_crop_for_ocr(aligned: np.ndarray, blank: np.ndarray, bbox: BBox,
                       new_ink: np.ndarray | None = None, pad: int = 8) -> np.ndarray:
    """Field crop with the template's printed lines/labels painted over and
    trimmed to the handwriting, so OCR reads only what the applicant wrote."""
    page_c = crop(aligned, bbox).copy()
    printed = cv2.dilate(ink_mask(crop(blank, bbox)), np.ones((3, 3), np.uint8))
    written = crop(new_ink, bbox) if new_ink is not None else \
        cv2.bitwise_and(ink_mask(page_c), cv2.bitwise_not(printed))
    paper = int(np.median(page_c[(printed == 0) & (written == 0)])) \
        if ((printed == 0) & (written == 0)).any() else 255
    page_c[(printed > 0) & (written == 0)] = paper
    # trim to the written part: handwriting models hallucinate on empty space
    ys, xs = np.where(written > 0)
    if len(xs) > 20:
        m = 6
        page_c = page_c[max(0, ys.min() - m):ys.max() + m + 1,
                        max(0, xs.min() - m):xs.max() + m + 1]
    # stretch contrast so paper is white and ink is dark
    page_c = cv2.normalize(page_c, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.copyMakeBorder(page_c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
