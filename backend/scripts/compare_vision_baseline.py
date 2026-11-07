"""Score FormSense against the OCR + regex baseline, field by field and issue by issue.

    # forms from the synthetic generator (images/ + labels/), admission form template
    python -m formsense.generator.generate --count 30 --split dev --seed 1 --out data/synthetic/dev
    python scripts/compare_vision_baseline.py --dir data/synthetic/dev --limit 10

    # the small demo form (needs python scripts/make_demo.py)
    python scripts/compare_vision_baseline.py --demo

Two kinds of folder work:
  - generator output: <dir>/images/*.jpg|png and <dir>/labels/<form_id>.json.
    Both systems' fields also go through the rules (rules/audit.py), so the
    issues they find are scored against the generator's true issues.
  - demo samples: <name>.jpg with <name>.truth.json next to it (fields only).

Writes outputs/comparison/summary.md, summary.json, fields.csv and issues.csv.
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.baseline import BaselineAuditor  # noqa: E402
from formsense.evaluation import find_forms, fmt  # noqa: E402
from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.scoring import score_field, score_issues, summarise, summarise_issues  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

DEMO_DIR = BACKEND / "data" / "templates" / "demo" / "samples"
DEMO_TPL = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
REAL_TPL = BACKEND / "data" / "templates" / "admission_form.json"
OUT = BACKEND / "outputs" / "comparison"
SYSTEMS = ("baseline", "formsense")
NAMES = {"baseline": "Baseline (OCR + regex)", "formsense": "FormSense"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="folder of forms (default: data/synthetic/dev)")
    ap.add_argument("--template", help="template JSON (default: admission form, or demo with --demo)")
    ap.add_argument("--demo", action="store_true", help="use the demo form samples")
    ap.add_argument("--limit", type=int, help="only the first N forms")
    ap.add_argument("--no-trocr", action="store_true", help="PaddleOCR only (faster)")
    args = ap.parse_args()

    folder = Path(args.dir) if args.dir else (DEMO_DIR if args.demo else BACKEND / "data" / "synthetic" / "dev")
    tpl_path = args.template or (DEMO_TPL if args.demo else REAL_TPL)
    template, blank = load_template(tpl_path)
    kinds = {f.name: f.type for f in template.fields}
    forms = find_forms(folder)[:args.limit] if args.limit else find_forms(folder)
    if not forms:
        sys.exit(f"No forms with ground truth found in {folder}.\n"
                 "Generate some: python -m formsense.generator.generate --count 30 --split dev --out data/synthetic/dev")
    with_issues = all(lab is not None for _, _, lab in forms)
    if with_issues:
        from formsense.rules.audit import audit

    ocr = FieldOCR(use_trocr=not args.no_trocr)
    baseline = BaselineAuditor(template, reader=ocr.paddle)   # same PaddleOCR model for both
    rows = {s: [] for s in SYSTEMS}
    issue_rows = {s: [] for s in SYSTEMS}
    secs = {s: 0.0 for s in SYSTEMS}
    detail, issue_detail = [], []

    for i, (img_path, truth, label) in enumerate(forms, 1):
        page = load_pages(img_path)[0]
        t0 = time.time()
        b_fields = baseline.run(page).fields
        t1 = time.time()
        v_fields = analyze_page(page, template, blank, ocr=ocr).fields
        t2 = time.time()
        secs["baseline"] += t1 - t0
        secs["formsense"] += t2 - t1
        preds = {"baseline": {r.name: r for r in b_fields}, "formsense": {r.name: r for r in v_fields}}
        for name, tr in truth.items():
            for s in SYSTEMS:
                r = score_field(kinds[name], preds[s][name], tr)
                rows[s].append(r)
                p = preds[s][name]
                detail.append({"form": img_path.stem, "field": name, "system": s, "correct": r["correct"],
                               "truth": tr.get("text", tr.get("checked", tr["status"])),
                               "truth_status": tr["status"],
                               "predicted": p.raw_text or p.value or p.status.value,
                               "predicted_status": p.status.value})
        line = f"[{i}/{len(forms)}] {img_path.name}: baseline {t1 - t0:.1f}s, formsense {t2 - t1:.1f}s"
        if with_issues:
            today = dt.date.fromisoformat(label["reference_date"])
            for s, fields in (("baseline", b_fields), ("formsense", v_fields)):
                res = score_issues(audit(fields, today=today).issues, label["issues"])
                res["truth_n"] = len(label["issues"])
                issue_rows[s].append(res)
                issue_detail.append({"form": img_path.stem, "system": s, "true_issues": len(label["issues"]),
                                     "found": res["tp"], "missed": " | ".join(res["missed"]),
                                     "false_alarms": " | ".join(res["extra"])})
            line += (f" | issues: true {len(label['issues'])}, FormSense found "
                     f"{issue_rows['formsense'][-1]['tp']} (+{issue_rows['formsense'][-1]['fp']} extra)")
        print(line)

    summary = {s: summarise(rows[s]) for s in SYSTEMS}
    if with_issues:
        for s in SYSTEMS:
            summary[s].update(summarise_issues(issue_rows[s]))
    for s in SYSTEMS:
        summary[s]["Seconds per form"] = round(secs[s] / len(forms), 1)

    n_issues = sum(len(lab["issues"]) for _, _, lab in forms) if with_issues else 0
    head = f"# FormSense vs. baseline ({len(forms)} forms, {len(rows['formsense'])} fields"
    head += f", {n_issues} true issues)" if with_issues else ")"
    lines = [head, "", f"| Metric | {NAMES['baseline']} | {NAMES['formsense']} |", "| --- | --- | --- |"]
    lines += [f"| {m} | {fmt(summary['baseline'].get(m))} | {fmt(summary['formsense'][m])} |"
              for m in summary["formsense"]]
    md = "\n".join(lines) + "\n"
    if with_issues:
        md += ("\nIssues are scored end to end: each system's fields go through the same rules "
               "(rules/audit.py); an issue counts when the same rule fires on the same fields.\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, data in (("fields.csv", detail), ("issues.csv", issue_detail)):
        if data:
            with open(OUT / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    print("\n" + md)
    wrong = [d for d in detail if d["system"] == "formsense" and not d["correct"]]
    if wrong:
        print("FormSense field mistakes (for error analysis):")
        for d in wrong[:20]:
            print(f"  {d['form']:<12} {d['field']:<18} truth={d['truth']!r:<24} ({d['truth_status']})"
                  f"  got={d['predicted']!r} ({d['predicted_status']})")
    if with_issues:
        misses = [d for d in issue_detail if d["system"] == "formsense" and (d["missed"] or d["false_alarms"])]
        if misses:
            print("\nFormSense issue mistakes:")
            for d in misses[:20]:
                print(f"  {d['form']:<12} missed: {d['missed'] or '-'}   extra: {d['false_alarms'] or '-'}")
    print(f"\nSaved {OUT.relative_to(BACKEND).as_posix()}/ summary.md, summary.json, fields.csv"
          + (", issues.csv" if with_issues else ""))


if __name__ == "__main__":
    main()
