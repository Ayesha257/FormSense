"""Score FormSense's vision step against the OCR + regex baseline.

    python scripts/compare_vision_baseline.py                  # demo samples
    python scripts/compare_vision_baseline.py --limit 5        # quick run
    python scripts/compare_vision_baseline.py --dir data/real/labelled --template data/templates/admission_form.json

Every <name>.jpg/.png in --dir needs a <name>.truth.json next to it
(scripts/make_demo.py writes these for the demo samples).

Writes outputs/comparison/summary.md, summary.json and fields.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.baseline import BaselineAuditor  # noqa: E402
from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.scoring import score_field, summarise  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

DEMO_DIR = BACKEND / "data" / "templates" / "demo" / "samples"
DEMO_TPL = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
OUT = BACKEND / "outputs" / "comparison"
SYSTEMS = ("baseline", "formsense")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEMO_DIR))
    ap.add_argument("--template", default=str(DEMO_TPL))
    ap.add_argument("--limit", type=int, help="only the first N forms")
    ap.add_argument("--no-trocr", action="store_true", help="PaddleOCR only (faster)")
    args = ap.parse_args()

    template, blank = load_template(args.template)
    kinds = {f.name: f.type for f in template.fields}
    images = sorted(p for p in Path(args.dir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and p.with_suffix(".truth.json").exists())
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"No images with .truth.json found in {args.dir}. Run: python scripts/make_demo.py")

    ocr = FieldOCR(use_trocr=not args.no_trocr)
    baseline = BaselineAuditor(template, reader=ocr.paddle)   # same PaddleOCR model for both
    rows = {s: [] for s in SYSTEMS}
    secs = {s: 0.0 for s in SYSTEMS}
    detail = []

    for i, img_path in enumerate(images, 1):
        truth = {t["name"]: t for t in json.loads(img_path.with_suffix(".truth.json").read_text())}
        page = load_pages(img_path)[0]
        t0 = time.time()
        b = {r.name: r for r in baseline.run(page).fields}
        t1 = time.time()
        v = {r.name: r for r in analyze_page(page, template, blank, ocr=ocr).fields}
        t2 = time.time()
        secs["baseline"] += t1 - t0
        secs["formsense"] += t2 - t1
        for name, tr in truth.items():
            for sysname, preds in (("baseline", b), ("formsense", v)):
                r = score_field(kinds[name], preds[name], tr)
                rows[sysname].append(r)
                detail.append({"form": img_path.stem, "field": name, "system": sysname,
                               "correct": r["correct"], "truth": tr.get("text", tr.get("checked", tr["status"])),
                               "predicted": preds[name].raw_text or preds[name].value or preds[name].status.value})
        print(f"[{i}/{len(images)}] {img_path.name}: baseline {t1 - t0:.1f}s, formsense {t2 - t1:.1f}s")

    summary = {s: summarise(rows[s]) for s in SYSTEMS}
    for s in SYSTEMS:
        summary[s]["Seconds per form"] = round(secs[s] / len(images), 1)

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# FormSense vs. baseline ({len(images)} forms, {len(rows['formsense'])} fields each system)",
             "", "| Metric | Baseline (OCR + regex) | FormSense (vision) |", "| --- | --- | --- |"]
    for metric in summary["formsense"]:
        fmt = lambda x: "n/a" if x is None else f"{x}"  # noqa: E731
        lines.append(f"| {metric} | {fmt(summary['baseline'][metric])} | {fmt(summary['formsense'][metric])} |")
    md = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "fields.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)

    print("\n" + md)
    wrong = [d for d in detail if d["system"] == "formsense" and not d["correct"]]
    if wrong:
        print("FormSense mistakes (for error analysis):")
        for d in wrong[:15]:
            print(f"  {d['form']:<22} {d['field']:<18} truth={d['truth']!r:<20} got={d['predicted']!r}")
    print(f"\nSaved {OUT.relative_to(BACKEND).as_posix()}/summary.md, summary.json, fields.csv")


if __name__ == "__main__":
    main()
