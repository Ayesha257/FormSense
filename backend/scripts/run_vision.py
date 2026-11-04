"""Run the vision step on one filled form and save the results.

    python scripts/run_vision.py path/to/form.jpg
    python scripts/run_vision.py data/templates/demo/samples/complete.jpg --demo
    python scripts/run_vision.py form.jpg --no-ocr      # detectors only (fast)

Writes outputs/<name>_annotated.png and outputs/<name>_fields.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.draw import annotate  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

DEMO = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
REAL = BACKEND / "data" / "templates" / "admission_form.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("form")
    ap.add_argument("--demo", action="store_true", help="use the demo template")
    ap.add_argument("--template", help="template JSON (default: admission form)")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-trocr", action="store_true", help="PaddleOCR only")
    args = ap.parse_args()

    tpl_path = Path(args.template) if args.template else (DEMO if args.demo else REAL)
    template, blank = load_template(tpl_path)
    ocr = None if args.no_ocr else FieldOCR(use_trocr=not args.no_trocr)

    page = load_pages(args.form)[0]
    t0 = time.time()
    out = analyze_page(page, template, blank, ocr=ocr)
    secs = time.time() - t0

    al = out.alignment
    print(f"\nAlignment: {'OK' if al.ok else 'FAILED (used plain resize)'}"
          f"  matches={al.matches} inliers={al.inliers}   time={secs:.1f}s\n")
    print(f"{'field':<20}{'status':<12}{'ink':>7}  {'conf':>5}  text / value")
    print("-" * 72)
    for r in out.fields:
        conf = f"{r.ocr_confidence:.2f}" if r.ocr_confidence is not None else "  -  "
        shown = r.raw_text or r.value or ""
        print(f"{r.name:<20}{r.status.value:<12}{r.ink_ratio:>7.3f}  {conf:>5}  {shown}")

    outdir = BACKEND / "outputs"
    outdir.mkdir(exist_ok=True)
    stem = Path(args.form).stem
    cv2.imwrite(str(outdir / f"{stem}_annotated.png"), annotate(out.aligned, out.fields))
    (outdir / f"{stem}_fields.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in out.fields], indent=2))
    print(f"\nSaved outputs/{stem}_annotated.png and outputs/{stem}_fields.json")


if __name__ == "__main__":
    main()
