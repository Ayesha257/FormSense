"""Check (and tune) the empty-box / tick / signature detectors. No OCR, so it is fast.

    python scripts/eval_detectors.py --dir data/synthetic/dev          # tune on dev
    python scripts/eval_detectors.py --dir data/synthetic/test         # report on test, never tune on it

For every field type it prints the accuracy with the current thresholds
(formsense/vision/fields.py) and the threshold that would be best on this set,
plus the forms where the detector was wrong.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.schema import FieldType  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision import fields as F  # noqa: E402

CURRENT = {"text": F.EMPTY_TEXT_MAX, "checkbox": F.CHECKBOX_MIN, "signature": F.SIGNATURE_MIN}
POSITIVE = {"text": "written", "checkbox": "ticked", "signature": "signed"}


def best_threshold(neg: list[float], pos: list[float]) -> tuple[float, float]:
    """Threshold t (positive when ratio >= t) with the highest accuracy on this set."""
    vals = sorted(set(neg + pos))
    cands = [0.0] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1e-6] if vals else [0.0]
    best = max(cands, key=lambda t: (sum(x < t for x in neg) + sum(x >= t for x in pos), -abs(t)))
    acc = (sum(x < best for x in neg) + sum(x >= best for x in pos)) / max(1, len(neg) + len(pos))
    return best, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/synthetic/dev")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    folder = BACKEND / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)
    labels = sorted((folder / "labels").glob("*.json"))[:args.limit]
    if not labels:
        sys.exit(f"No generator labels in {folder}/labels")
    template, blank = load_template()
    kinds = {f.name: ("text" if f.type in (FieldType.TEXT, FieldType.DATE) else f.type.value)
             for f in template.fields}
    ratios = defaultdict(lambda: {"neg": [], "pos": []})
    wrong = []
    align_fail = 0
    for n, lab_path in enumerate(labels, 1):
        lab = json.loads(lab_path.read_text(encoding="utf-8"))
        page = load_pages(folder / "images" / lab["image"])[0]
        out = analyze_page(page, template, blank, ocr=None)
        align_fail += not out.alignment.ok
        truth = {f["name"]: f for f in lab["fields"]}
        for r in out.fields:
            k, t = kinds[r.name], truth[r.name]
            positive = t["checked"] if k == "checkbox" else t["status"] != "missing"
            ratios[k]["pos" if positive else "neg"].append(r.ink_ratio)
            predicted = r.ink_ratio >= CURRENT[k] if k != "text" else r.ink_ratio >= CURRENT[k]
            if predicted != positive:
                wrong.append(f"{lab['form_id']} ({lab['capture']['profile']}): {r.name} "
                             f"truth={'yes' if positive else 'no'} ink={r.ink_ratio}")
        print(f"\r{n}/{len(labels)} forms", end="", flush=True)
    print(f"\n\n{len(labels)} forms, alignment failed on {align_fail}\n")
    print(f"{'type':<10}{'n':>5}{'accuracy now':>14}{'threshold now':>15}{'best':>9}{'best acc':>10}"
          f"   {'not ' + 'x':<6} ink p50/p95 | x ink p5/p50")
    for k in ("text", "checkbox", "signature"):
        neg, pos = ratios[k]["neg"], ratios[k]["pos"]
        if not neg and not pos:
            continue
        now = (sum(x < CURRENT[k] for x in neg) + sum(x >= CURRENT[k] for x in pos)) / (len(neg) + len(pos))
        best, acc = best_threshold(neg, pos)
        q = lambda a, p: f"{np.percentile(a, p):.4f}" if a else "-"  # noqa: E731
        print(f"{k:<10}{len(neg) + len(pos):>5}{100 * now:>13.1f}%{CURRENT[k]:>15.4f}{best:>9.4f}{100 * acc:>9.1f}%"
              f"   {q(neg, 50)}/{q(neg, 95)} | {q(pos, 5)}/{q(pos, 50)}   ({POSITIVE[k]}: {len(pos)})")
    if wrong:
        print(f"\nWrong with current thresholds ({len(wrong)}):")
        for w in wrong[:40]:
            print("  " + w)


if __name__ == "__main__":
    main()
