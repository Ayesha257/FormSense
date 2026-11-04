"""Load the form template (field boxes) and its blank image."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from formsense.schema import FormTemplate

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = BACKEND_DIR / "data" / "templates" / "admission_form.json"


def load_template(path: str | Path = DEFAULT_TEMPLATE) -> tuple[FormTemplate, np.ndarray]:
    """Return (template, blank grayscale image)."""
    path = Path(path)
    template = FormTemplate(**json.loads(path.read_text(encoding="utf-8")))
    img_path = Path(template.image_path)
    if not img_path.is_absolute():
        img_path = BACKEND_DIR / img_path
    blank = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if blank is None:
        raise FileNotFoundError(
            f"Blank template image not found: {img_path}. "
            "Export the blank form as PNG and put it there."
        )
    unmarked = [f.name for f in template.fields if f.bbox[2] == 0 or f.bbox[3] == 0]
    if unmarked:
        raise ValueError(
            f"{len(unmarked)} field(s) have no box yet (e.g. {unmarked[0]}). "
            "Run: python scripts/mark_fields.py"
        )
    return template, blank
