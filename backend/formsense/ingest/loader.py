"""Load a PDF or image into OpenCV page images and clean them up.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # older PyMuPDF
    import fitz

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_pages(path: str | Path, dpi: int = 200) -> list[np.ndarray]:
    """Return a list of BGR images, one per page."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pages.append(img)
        return pages
    if ext in IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return [img]
    raise ValueError(f"Unsupported file type: {ext}")


def load_bytes(data: bytes, filename: str, dpi: int = 200) -> list[np.ndarray]:
    """Same as load_pages but for uploaded bytes (used by Streamlit)."""
    if filename.lower().endswith(".pdf"):
        pages = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                code = cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
                pages.append(cv2.cvtColor(img, code))
        return pages
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {filename}")
    return [img]


def _rotate(img: np.ndarray, angle: float, border=cv2.BORDER_REPLICATE) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=border)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """Projection-profile method: the rotation that makes text rows sharpest wins."""
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        rows = _rotate(ink, angle, cv2.BORDER_CONSTANT).sum(axis=1, dtype=np.float64)
        score = float(np.var(rows))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(gray: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate a grayscale page so text lines are horizontal."""
    angle = estimate_skew(gray, max_angle)
    return gray if abs(angle) < 0.25 else _rotate(gray, angle)


def preprocess(img: np.ndarray) -> np.ndarray:
    """BGR page -> cleaned grayscale page (denoised + deskewed)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return deskew(gray)
