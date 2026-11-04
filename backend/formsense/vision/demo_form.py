"""A small fake admission form for tests and demos (until the real form exists)."""
from __future__ import annotations

import cv2
import numpy as np

from formsense.schema import FieldSpec, FieldType, FormTemplate

W, H = 1240, 1754   # A4 at 150 dpi
INK = (20, 20, 120)  # dark blue pen (BGR)

# name, label, type, bbox
FIELDS = [
    ("full_name", "Full Name", FieldType.TEXT, (330, 260, 760, 70)),
    ("father_name", "Father's Name", FieldType.TEXT, (330, 360, 760, 70)),
    ("cnic", "CNIC No.", FieldType.TEXT, (330, 460, 760, 70)),
    ("date_of_birth", "Date of Birth", FieldType.DATE, (330, 560, 400, 70)),
    ("program_bscs", "BS Computer Science", FieldType.CHECKBOX, (120, 720, 44, 44)),
    ("program_bsse", "BS Software Engineering", FieldType.CHECKBOX, (120, 800, 44, 44)),
    ("program_bsai", "BS Artificial Intelligence", FieldType.CHECKBOX, (120, 880, 44, 44)),
    ("declaration_date", "Date", FieldType.DATE, (330, 1300, 400, 70)),
    ("signature", "Signature", FieldType.SIGNATURE, (330, 1420, 500, 140)),
]


def template() -> FormTemplate:
    return FormTemplate(
        name="test_form", image_path="", width=W, height=H,
        fields=[FieldSpec(name=n, label=l, type=t, bbox=b,
                          group="program" if n.startswith("program") else None)
                for n, l, t, b in FIELDS],
    )


def blank_form() -> np.ndarray:
    img = np.full((H, W), 255, np.uint8)
    cv2.putText(img, "UNIVERSITY ADMISSION FORM 2026", (190, 120),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, 0, 3)
    cv2.rectangle(img, (60, 40), (W - 60, H - 40), 0, 3)
    cv2.putText(img, "Section A: Personal Information", (80, 215), cv2.FONT_HERSHEY_DUPLEX, 0.9, 0, 2)
    cv2.putText(img, "Section B: Program (tick one)", (80, 690), cv2.FONT_HERSHEY_DUPLEX, 0.9, 0, 2)
    cv2.putText(img, "Section C: Declaration", (80, 1150), cv2.FONT_HERSHEY_DUPLEX, 0.9, 0, 2)
    cv2.putText(img, "I declare that the information given above is correct.", (80, 1220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2)
    for n, label, t, (x, y, w, h) in FIELDS:
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
        if t == FieldType.CHECKBOX:
            cv2.putText(img, label, (x + 70, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        else:
            cv2.putText(img, label, (80, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return img


def filled_form(values: dict[str, str], ticks: set[str], signed: bool) -> np.ndarray:
    img = cv2.cvtColor(blank_form(), cv2.COLOR_GRAY2BGR)
    boxes = {n: b for n, _, _, b in FIELDS}
    for name, text in values.items():
        x, y, w, h = boxes[name]
        cv2.putText(img, text, (x + 15, y + h - 18), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1.3, INK, 2, cv2.LINE_AA)
    for name in ticks:
        x, y, w, h = boxes[name]
        pts = np.array([(x + 8, y + 22), (x + 18, y + 36), (x + 38, y + 8)], np.int32)
        cv2.polylines(img, [pts], False, INK, 4, cv2.LINE_AA)
    if signed:
        x, y, w, h = boxes["signature"]
        t = np.linspace(0, 4 * np.pi, 200)
        xs = x + 40 + t / (4 * np.pi) * (w - 80)
        ys = y + h / 2 + 30 * np.sin(t * 1.5) * np.cos(t * 0.4)
        cv2.polylines(img, [np.int32(np.c_[xs, ys])], False, INK, 3, cv2.LINE_AA)
    return img


def phone_photo(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simulate a phone photo: perspective tilt, rotation, uneven light, noise, bigger canvas."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    canvas_w, canvas_h = int(w * 1.25), int(h * 1.2)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.04, 0.04, (4, 2)) * [w, h]
    off = np.float32([(canvas_w - w) / 2, (canvas_h - h) / 2])
    dst = np.float32(src + off + jitter)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (canvas_w, canvas_h), borderValue=(170, 160, 150))
    grad = np.linspace(0.75, 1.0, canvas_w)[None, :, None]
    out = np.clip(out * grad + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)
    return out
