"""The real evaluation: forms as IMAGES through the whole system, scored, with an error breakdown.

    python scripts/evaluate_all.py --dir data/synthetic/test                 # FormSense + baseline
    python scripts/evaluate_all.py --dir data/synthetic/test --limit 20      # quick look
    python scripts/evaluate_all.py --systems formsense --reuse               # re-score from saved readings
    python scripts/evaluate_all.py --systems formsense,baseline,vlm          # also the Groq VLM (needs a key)
    python scripts/evaluate_all.py --dry-run --limit 10                      # test the chain, simulated OCR

    # score the readings saved by `python -m formsense.eval.run_comparison` (no OCR run needed)
    python scripts/evaluate_all.py --dir data/synthetic/test --readings data/eval_results --sweep-threshold

    # choose the OCR-confidence cutoff for "unreadable" from data (run on DEV, then report on TEST)
    python scripts/evaluate_all.py --dir data/synthetic/dev --systems formsense --sweep-threshold

Why this exists. `formsense.eval.harness.evaluate_dataset` feeds the language side PERFECT text
straight from the labels. That is a CEILING (it always looks good, and it cannot see unreadable
handwriting or tick marks). The real figure needs the images to go through alignment, ink detection
and OCR first. This script does that, scores each system with the same metrics (formsense/eval/metrics.py,
formsense/eval/harness.py), and shows where the lost accuracy went (formsense/eval/failure_analysis.py).

Output, in outputs/evaluation/:
  summary.md        tables for the slides (headline, per field, per rule, examples, worst forms)
  summary.json      the same numbers for scripts
  field_errors.csv  every field of every form with its outcome
  issue_errors.csv  every missed or false issue and what caused it
  predictions/      each system's raw readings, so --reuse can re-score without running OCR again

Needs labelled images (images/ + labels/ from formsense.generator.generate).
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.eval.failure_analysis import analyse, format_markdown  # noqa: E402
from formsense.eval.harness import (LabelledForm, aggregate_results, evaluate_dataset,  # noqa: E402
                                    full_pipeline_system)
from formsense.evaluation import write_csv  # noqa: E402
from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.nlp.template_info import TemplateInfo  # noqa: E402
from formsense.rag.retriever import field_labels  # noqa: E402
from formsense.rules.audit import audit  # noqa: E402
from formsense.schema import FieldResult, FieldType, Issue  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402

TEMPLATE = BACKEND / "data" / "templates" / "admission_form.json"
OUT = BACKEND / "outputs" / "evaluation"
NAMES = {"ceiling": "Ceiling (perfect text)", "baseline": "Baseline (OCR + regex)", "formsense": "FormSense (images)",
         "vlm": "VLM (Groq)"}
SYSTEMS = ("formsense", "baseline", "vlm")
READINGS_PREFIX = {"formsense": "full_", "baseline": "baseline_"}     # file names written by run_comparison


def num(x, nd=3):
    return "n/a" if x != x or x is None else f"{x:.{nd}f}"


def text_accuracy(result, text_fields):
    """Exact-match accuracy over written fields only. The brief's field metric compares text, and tick boxes
    and the signature have none (empty == empty), so they always count as correct and inflate the headline."""
    per = result.field_extraction.per_field
    n = sum(per[f].n for f in text_fields if f in per)
    return sum(per[f].exact_matches for f in text_fields if f in per) / n if n else float("nan")


def prf(p):
    return f"{num(p.precision, 2)} / {num(p.recall, 2)} / {num(p.f1, 2)}"


def load_forms(folder: Path) -> list[tuple[Path, LabelledForm]]:
    labels = sorted((folder / "labels").glob("*.json"))
    if not labels:
        sys.exit(f"No labelled forms in {folder}. Generate some:\n"
                 "  python -m formsense.generator.generate --count 60 --split test --seed 100000 --out data/synthetic/test")
    out = []
    for lab in labels:
        d = json.loads(lab.read_text(encoding="utf-8"))
        img = folder / "images" / d["image"]
        if img.exists():
            out.append((img, LabelledForm(d.get("form_id", lab.stem), [FieldResult(**f) for f in d["fields"]],
                                          [Issue(**i) for i in d["issues"]],
                                          dt.date.fromisoformat(d["reference_date"]))))
    return out


def simulated_texts(form: LabelledForm, template) -> list[str]:
    """True texts of the non-empty text fields, in template order (what SimulatedOCR hands out)."""
    by = {f.name: f for f in form.fields}
    return [by[s.name].raw_text or "" for s in template.fields
            if s.type not in (FieldType.CHECKBOX, FieldType.SIGNATURE) and by[s.name].status.value != "missing"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(BACKEND / "data" / "synthetic" / "test"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--systems", default="formsense,baseline", help=f"comma separated, from: {', '.join(SYSTEMS)}")
    ap.add_argument("--readings", help="folder with saved readings from formsense.eval.run_comparison "
                                       "(full_<form>.json -> formsense, baseline_<form>.json -> baseline); "
                                       "scored without running any OCR")
    ap.add_argument("--reuse", action="store_true", help="use saved readings from outputs/evaluation/predictions")
    ap.add_argument("--sweep-threshold", action="store_true",
                    help="also re-score FormSense's readings at cutoffs 0.30-0.90 for 'unreadable' (tune on dev)")
    ap.add_argument("--no-trocr", action="store_true", help="PaddleOCR only (faster)")
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between VLM calls")
    ap.add_argument("--dry-run", action="store_true", help="simulated OCR from the labels: tests the chain only")
    ap.add_argument("--noise", type=float, default=0.12, help="dry-run: share of misread fields")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.dry_run:
        systems = ["formsense"]
        print("*** DRY RUN: OCR is simulated from the ground truth. These numbers are NOT results. ***\n")
    bad = [s for s in systems if s not in SYSTEMS]
    if bad or not systems:
        sys.exit(f"Unknown system(s) {bad}. Choose from: {', '.join(SYSTEMS)}")
    out_dir = Path(args.out + ("_dryrun" if args.dry_run else ""))
    folder = Path(args.dir)
    forms = load_forms(folder)
    forms = forms[:args.limit] if args.limit else forms
    template, blank = load_template(TEMPLATE)
    kinds = {f.name: f.type for f in template.fields}
    text_fields = [f.name for f in template.fields if f.type not in (FieldType.CHECKBOX, FieldType.SIGNATURE)]
    labels = field_labels(TEMPLATE)
    for spec in template.fields:               # tick boxes share a group label: show which box
        if spec.type == FieldType.CHECKBOX:
            labels[spec.name] = f"{labels.get(spec.name, spec.group)}: {spec.label}"
    tpl_info = TemplateInfo.load(TEMPLATE)

    ocr = baseline = vlm = None
    if not args.dry_run:
        if any(s in systems for s in ("formsense", "baseline")) and not (args.reuse or args.readings):
            from formsense.vision.ocr import FieldOCR
            ocr = FieldOCR(use_trocr=not args.no_trocr)
        if "baseline" in systems and not (args.reuse or args.readings):
            from formsense.baseline import BaselineAuditor
            baseline = BaselineAuditor(template, reader=ocr.paddle)
        if "vlm" in systems and not args.reuse:
            from formsense.vlm import VLMAuditor
            vlm = VLMAuditor(template)
            vlm.client  # fail now if the key is missing

    pred_dir = out_dir / "predictions"
    results = {}
    seconds = {s: 0.0 for s in systems}
    failed = []
    outputs = {s: [] for s in systems}      # (LabelledForm, audit result)
    raws = {s: {} for s in systems}         # form id -> raw readings, for the threshold sweep
    for i, (img, form) in enumerate(forms, 1):
        page = None
        for s in systems:
            cache = pred_dir / s / f"{img.stem}.json"
            saved = Path(args.readings) / f"{READINGS_PREFIX[s]}{form.form_id}.json" if args.readings and s in READINGS_PREFIX else None
            t0 = time.time()
            try:
                if saved is not None and saved.exists():
                    raw = [FieldResult(**d) for d in json.loads(saved.read_text(encoding="utf-8"))]
                elif args.reuse and cache.exists():
                    raw = [FieldResult(**d) for d in json.loads(cache.read_text())]
                else:
                    if args.reuse:
                        raise FileNotFoundError(f"no saved readings for {img.stem} ({s}); run without --reuse first")
                    page = page if page is not None else load_pages(img)[0]
                    if s == "formsense":
                        from formsense.eval.simulated_ocr import SimulatedOCR
                        use = SimulatedOCR(simulated_texts(form, template), args.noise, seed=i) if args.dry_run else ocr
                        raw = analyze_page(page, template, blank, ocr=use).fields
                    elif s == "baseline":
                        raw = baseline.run(page).fields
                    else:
                        raw = vlm.run(page).fields
                        if args.delay > 0 and i < len(forms):
                            time.sleep(args.delay)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps([r.model_dump(mode="json") for r in raw]), encoding="utf-8")
                raws[s][form.form_id] = raw
                outputs[s].append((form, audit(raw, template=TEMPLATE, today=form.reference_date)))
            except Exception as e:                       # one bad form must not stop a long run
                failed.append((img.name, s, str(e)))
                print(f"  ! {img.name} ({s}): {e}")
            seconds[s] += time.time() - t0
        print(f"[{i}/{len(forms)}] {img.name}")

    # only forms every selected system handled, so no system is helped or hurt by another's failure
    ok_ids = set.intersection(*[{f.form_id for f, _ in outputs[s]} for s in systems]) if systems else set()
    if not ok_ids:
        sys.exit("No form was processed by every system; nothing to score. See the errors above.")
    for s in systems:
        outputs[s] = [(f, o) for f, o in outputs[s] if f.form_id in ok_ids]
        results[s] = aggregate_results(outputs[s])
    n_used = len(ok_ids)
    ceiling = None
    if (folder / "labels").is_dir():
        ceiling = evaluate_dataset(folder / "labels", full_pipeline_system(tpl_info),
                                   limit=len(forms) if args.limit else None)
        results = {"ceiling": ceiling, **results}

    # ---- markdown ----
    L = []
    if args.dry_run:
        L += ["# DRY RUN: simulated OCR, not results", ""]
    L += [f"# Evaluation: {n_used} forms from `{folder.name}`", "",
          "| System | Field accuracy | Text-field accuracy | CER | Missing-field P / R / F1 | Issue P / R / F1 | Seconds per form |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        fe = r.field_extraction
        sec = "n/a" if name == "ceiling" or args.reuse or args.readings else f"{seconds[name] / max(1, len(forms)):.1f}"
        L.append(f"| {NAMES[name]} | {num(fe.accuracy)} | {num(text_accuracy(r, text_fields))} | {num(fe.mean_cer)} | "
                 f"{prf(r.missing_detection)} | "
                 f"{prf(r.issue_detection)} | {sec} |")
    L += ["", "Field accuracy is the brief's metric. It counts tick boxes and the signature, which have no text to compare, "
              "as always correct, so Text-field accuracy (written fields only) is the honest figure for reading text.", "",
          "The ceiling row gives the rules perfect text (it is the best the language side can do, and it cannot "
              "see unreadable handwriting); every other row starts from the images. The gap between the ceiling "
              "and a system is the accuracy lost before the rules run: alignment, ink detection and OCR.", "",
          "### Issue detection by type (P / R / F1)", "",
          "| Type | " + " | ".join(NAMES[n] for n in results) + " |",
          "| --- | " + " | ".join("---" for _ in results) + " |"]
    for t in ("missing", "unreadable", "format", "inconsistency", "selection"):
        L.append(f"| {t} | " + " | ".join(prf(r.issue_detection_by_type[t]) for r in results.values()) + " |")
    L += ["", "### Missing vs. unreadable (rows = truth, columns = predicted)", ""]
    for name, r in results.items():
        cm = r.missing_vs_unreadable
        L += [f"**{NAMES[name]}**", "", "| truth \\ predicted | " + " | ".join(cm.labels) + " |",
              "| --- | " + " | ".join("---" for _ in cm.labels) + " |"]
        for lab, row in zip(cm.labels, cm.as_rows()):
            L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
        L.append("")

    analyses = {}
    for s in systems:
        analyses[s] = analyse(outputs[s], kinds)
        L += [f"# {NAMES[s]}", "", format_markdown(analyses[s], labels)]
    sweep_rows = []
    if args.sweep_threshold and "formsense" in systems:
        from formsense.eval import threshold as TH
        sweep_rows = TH.sweep([(f, raws["formsense"][f.form_id]) for f, _ in outputs["formsense"]], TEMPLATE)
        L += ["", TH.format_markdown(sweep_rows)]
    if failed:
        L += ["", f"{len(failed)} failure(s), excluded:"] + [f"- {n} ({s}): {e}" for n, s, e in failed]
    md = "\n".join(L) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        {"forms": n_used, "dry_run": args.dry_run,
         "systems": {n: r.as_dict() for n, r in results.items()},
         "threshold_sweep": sweep_rows,
         "errors": {s: {"outcomes": dict(a.outcome_totals), "issue_errors": a.n_issue_errors,
                        "issue_errors_from_misread_fields": a.n_issue_errors_upstream,
                        "worst_forms": a.worst_forms[:10]} for s, a in analyses.items()}}, indent=2, default=str),
        encoding="utf-8")
    for s, a in analyses.items():
        write_csv(out_dir / f"field_errors_{s}.csv", a.field_rows)
        write_csv(out_dir / f"issue_errors_{s}.csv", a.issue_rows)
    print("\n" + md)
    print(f"Saved to {out_dir}: summary.md, summary.json, field_errors_<system>.csv, issue_errors_<system>.csv")


if __name__ == "__main__":
    main()
