"""Helpers shared by the evaluation scripts (forms with ground truth, result files)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from formsense.scoring import truth_from_label


def find_forms(folder: Path):
    """[(image path, truth {field: row}, label dict or None)] for a folder of forms.

    Generator output (images/ + labels/) carries true issues; demo samples
    (<name>.jpg + <name>.truth.json) carry field truth only.
    """
    folder = Path(folder)
    if (folder / "labels").is_dir():
        out = []
        for lab in sorted((folder / "labels").glob("*.json")):
            label = json.loads(lab.read_text(encoding="utf-8"))
            img = folder / "images" / label["image"]
            if img.exists():
                out.append((img, truth_from_label(label), label))
        return out
    return [(p, {t["name"]: t for t in json.loads(p.with_suffix(".truth.json").read_text())}, None)
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.with_suffix(".truth.json").exists()]


def fmt(x) -> str:
    return "n/a" if x is None else f"{x}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
