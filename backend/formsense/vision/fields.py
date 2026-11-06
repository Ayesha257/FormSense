"""Crop field boxes and measure the handwriting ("new ink") inside them.

Key idea: compare the aligned page with the blank template. Anything dark on
the page that is NOT already printed on the template is the applicant's ink.
That removes box borders, labels and guide lines without hand-tuning.
"""
from __future__ import annotations

import cv2
import numpy as np

from formsense.schema import BBox

# Tunable thresholds: share of the box area covered by new ink.
# Tuned on generator dev forms with scripts/eval_detectors.py (never on the test split).
EMPTY_TEXT_MAX = 0.0002    # text/date box below this = nothing written (~9 px of ink)
CHECKBOX_MIN = 0.06        # checkbox above this = ticked
SIGNATURE_MIN = 0.008      # signature box above this = signed

# How much darker than the surrounding paper a pixel must be to count as ink
INK_DARKNESS = 0.13        # pen
FAINT_DARKNESS = 0.08      # signature boxes: light pencil still counts
PRINTED_DARKNESS = 0.25    # what counts as printed on the blank template


def darkness(gray: np.ndarray) -> np.ndarray:
    """0..1 per pixel: how much darker than the local paper it is.

    The paper colour is estimated with a morphological closing at 1/4 scale
    (removes every stroke narrower than ~70 px), so shadows, grey phone
    photos and faded scans all end up on the same scale."""
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    bg = cv2.GaussianBlur(cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 3)
    d = 1.0 - gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)
    return np.clip(d, 0.0, 1.0)


def ink_mask(gray: np.ndarray, min_darkness: float = INK_DARKNESS) -> np.ndarray:
    """Binary mask (255 = ink), independent of lighting and paper colour."""
    return ((darkness(gray) > min_darkness) * 255).astype(np.uint8)


def _remove_border_remnants(mask: np.ndarray, printed: np.ndarray, length: int = 25,
                            near_px: int = 8) -> np.ndarray:
    """Delete thin straight lines lying next to printed lines: the left-over edge of a
    box border when alignment is a few pixels off. Solid blobs (a filled-in checkbox)
    and handwriting away from the borders are kept."""
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    near = cv2.dilate(printed, np.ones((2 * near_px + 1,) * 2, np.uint8))
    lines = cv2.bitwise_and(cv2.bitwise_or(horiz, vert), near)
    lines = cv2.bitwise_and(cv2.dilate(lines, np.ones((3, 3), np.uint8)),
                            cv2.bitwise_not(cv2.dilate(solid, np.ones((3, 3), np.uint8))))
    return cv2.bitwise_and(mask, cv2.bitwise_not(lines))


def new_ink_mask(aligned: np.ndarray, blank: np.ndarray, tolerance_px: int = 3,
                 min_darkness: float = INK_DARKNESS, page_dark: np.ndarray | None = None) -> np.ndarray:
    """Ink on the filled page that isn't printed on the blank template."""
    d = darkness(aligned) if page_dark is None else page_dark
    page_ink = ((d > min_darkness) * 255).astype(np.uint8)
    printed_raw = ((darkness(blank) > PRINTED_DARKNESS) * 255).astype(np.uint8)
    # grow printed strokes a little so small alignment errors don't count as ink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    printed = cv2.dilate(printed_raw, k)
    new = cv2.bitwise_and(page_ink, cv2.bitwise_not(printed))
    new = _remove_border_remnants(new, printed_raw)
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


def crop_pad(img: np.ndarray, x: int, y: int, w: int, h: int, fill=0) -> np.ndarray:
    """img[y:y+h, x:x+w], padded with `fill` where the window leaves the image."""
    H, W = img.shape[:2]
    out = np.full((h, w) + img.shape[2:], fill, dtype=img.dtype)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def local_shift(page_dark: np.ndarray, blank_dark: np.ndarray, bbox: BBox,
                search: int = 14, margin: int = 10, min_score: float = 0.3) -> tuple[int, int]:
    """Fine-tune one field's position after the page-level alignment.

    The homography can be a few pixels off in parts of a blurry or bent photo.
    Matching the printed box (and its surroundings) from the blank template inside a
    small window of the page finds where the box really is. Returns (dx, dy)."""
    x, y, w, h = bbox
    tmpl = crop_pad(blank_dark, x - margin, y - margin, w + 2 * margin, h + 2 * margin)
    if float(tmpl.std()) < 0.02:          # nothing printed around this box to match
        return 0, 0
    area = crop_pad(page_dark, x - margin - search, y - margin - search,
                    w + 2 * (margin + search), h + 2 * (margin + search))
    res = cv2.matchTemplate(area.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < min_score:
        return 0, 0
    return loc[0] - search, loc[1] - search


def field_ink(page_dark: np.ndarray, blank_dark: np.ndarray, bbox: BBox, shift: tuple[int, int],
              min_darkness: float = INK_DARKNESS, tolerance_px: int = 3) -> np.ndarray:
    """Applicant's ink inside one field (mask the size of the box).

    The page is read at the shifted position, the printed template at the template
    position, so box borders and labels cancel out even when alignment is off."""
    x, y, w, h = bbox
    dx, dy = shift
    page_ink = ((crop_pad(page_dark, x + dx, y + dy, w, h) > min_darkness) * 255).astype(np.uint8)
    printed_raw = ((crop_pad(blank_dark, x, y, w, h) > PRINTED_DARKNESS) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    new = cv2.bitwise_and(page_ink, cv2.bitwise_not(cv2.dilate(printed_raw, k)))
    new = _remove_border_remnants(new, printed_raw)
    return cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def mask_ratio(mask: np.ndarray, inset: int = 2) -> float:
    m = mask[inset:mask.shape[0] - inset, inset:mask.shape[1] - inset] if inset else mask
    return float((m > 0).mean()) if m.size else 0.0


def clean_crop_for_ocr(aligned: np.ndarray, blank: np.ndarray, bbox: BBox,
                       written: np.ndarray | None = None, shift: tuple[int, int] = (0, 0),
                       pad: int = 8) -> np.ndarray:
    """Field crop with the template's printed lines/labels painted over and trimmed
    to the handwriting, so OCR reads only what the applicant wrote.

    `written` is the field's ink mask from field_ink(); `shift` its local alignment."""
    x, y, w, h = bbox
    page_c = crop_pad(aligned, x + shift[0], y + shift[1], w, h, fill=255).copy()
    printed = cv2.dilate(ink_mask(crop_pad(blank, x, y, w, h, fill=255)), np.ones((3, 3), np.uint8))
    if written is None:
        written = cv2.bitwise_and(ink_mask(page_c), cv2.bitwise_not(printed))
    free = (printed == 0) & (written == 0)
    paper = int(np.median(page_c[free])) if free.any() else 255
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


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int):
    """Write patch into canvas at (x, y), clipped to the canvas (in place)."""
    H, W = canvas.shape[:2]
    h, w = patch.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch[y0 - y:y1 - y, x0 - x:x1 - x])
