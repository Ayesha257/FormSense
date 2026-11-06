"""Suggest ink thresholds from labelled forms.

    python -m formsense.nlp.calibrate data/synthetic/dev/labels

Reads every label JSON, compares ink_ratio between fields the generator marked empty and
fields it marked filled (separately for text boxes, signatures and checkboxes) and prints
numbers to copy into ExtractionConfig. Run it again whenever the template, the fonts or
the capture simulation change, and once more on the real scanned forms.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from formsense.nlp.template_info import TemplateInfo


def _mid(low_max: float, high_min: float) -> Optional[float]:
    """Geometric midpoint between the two clusters, or None if they overlap."""
    if low_max >= high_min:
        return None
    if low_max <= 0:
        return round(high_min / 2, 5)
    return round((low_max * high_min) ** 0.5, 5)


def collect(label_dir: Path, tpl: TemplateInfo) -> Dict[str, Dict[str, List[float]]]:
    groups: Dict[str, Dict[str, List[float]]] = {
        "text": {"empty": [], "filled": []},
        "signature": {"empty": [], "filled": []},
        "checkbox": {"empty": [], "filled": []},
    }
    for p in sorted(label_dir.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for f in doc.get("fields", []):
            info = tpl.fields.get(f["name"])
            if info is None:
                continue
            ink = float(f.get("ink_ratio") or 0.0)
            if info.type == "checkbox":
                key = "filled" if f.get("checked") else "empty"
                groups["checkbox"][key].append(ink)
            elif info.type == "signature":
                groups["signature"]["empty" if f.get("status") == "missing" else "filled"].append(ink)
            elif f.get("status") == "missing":
                groups["text"]["empty"].append(ink)
            elif f.get("status") == "ok":
                groups["text"]["filled"].append(ink)
    return groups


def suggest(groups: Dict[str, Dict[str, List[float]]]) -> Dict[str, Optional[float]]:
    def pair(g):
        e, f = groups[g]["empty"], groups[g]["filled"]
        return (max(e), min(f)) if e and f else None

    out: Dict[str, Optional[float]] = {}
    for key, g in (("ink_empty_below", "text"), ("signature_ink_min", "signature"), ("checkbox_ink_min", "checkbox")):
        p = pair(g)
        out[key] = _mid(*p) if p else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label_dir")
    ap.add_argument("--template")
    args = ap.parse_args(argv)
    tpl = TemplateInfo.load(args.template)
    groups = collect(Path(args.label_dir), tpl)
    for g, d in groups.items():
        for k, v in d.items():
            if v:
                print(f"{g:9} {k:7} n={len(v):5}  min={min(v):.5f}  median={statistics.median(v):.5f}  max={max(v):.5f}")
    print("\nSuggested ExtractionConfig values (None means the clusters overlap: inspect by hand):")
    for k, v in suggest(groups).items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
