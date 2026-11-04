"""Draw coloured field boxes on the aligned page for the UI / debugging."""
from __future__ import annotations

import cv2
import numpy as np

from formsense.schema import FieldResult, FieldStatus

# BGR colours
COLORS = {
    FieldStatus.OK: (60, 170, 60),          # green
    FieldStatus.MISSING: (40, 40, 220),     # red
    FieldStatus.UNREADABLE: (0, 150, 255),  # orange
    FieldStatus.INVALID: (180, 60, 200),    # purple
}
CHECKED = (200, 120, 0)     # blue
UNCHECKED = (150, 150, 150)  # gray


def annotate(aligned_gray: np.ndarray, results: list[FieldResult],
             show_labels: bool = True) -> np.ndarray:
    img = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR) if aligned_gray.ndim == 2 \
        else aligned_gray.copy()
    thick = max(2, img.shape[1] // 600)
    scale = max(0.4, img.shape[1] / 2400)
    for r in results:
        x, y, w, h = r.bbox
        if r.checked is not None:
            color = CHECKED if r.checked else UNCHECKED
        else:
            color = COLORS[r.status]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        if show_labels:
            label = r.name if r.checked is None else f"{r.name}: {r.value}"
            if r.status != FieldStatus.OK and r.checked is None:
                label += f" [{r.status.value}]"
            cv2.putText(img, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, max(1, thick - 1), cv2.LINE_AA)
    return img
