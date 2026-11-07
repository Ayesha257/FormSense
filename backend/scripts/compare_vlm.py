"""Stretch goal: baseline vs. VLM vs. FormSense on the same forms.

    # needs GROQ_API_KEY in backend/.env; answers are cached, so re-runs are free
    python scripts/compare_vlm.py --dir data/synthetic/test --limit 20

    # check the whole script without an API key (fake VLM built from the ground truth:
    # the numbers are NOT results, the output goes to outputs/comparison_vlm_dryrun/)
    python scripts/compare_vlm.py --dir data/synthetic/test --limit 5 --dry-run

    # only the VLM (fast, no OCR models), or any mix
    python scripts/compare_vlm.py --dir data/synthetic/test --systems vlm
    python scripts/compare_vlm.py --systems baseline,formsense,vlm --delay 3

The three systems all produce FieldResult lists, which go through the same rules
(rules/audit.py) and the same scoring (scoring.py). Metrics are computed on the
forms where every selected system answered, so a VLM failure never flatters or
punishes another system. Failures are listed separately.

Writes outputs/comparison_vlm/ : summary.md, summary.json, fields.csv, issues.csv.
Free-tier note: Groq allows roughly 30 requests per minute; --delay (seconds
between VLM calls, default 2.5) keeps you under it.
"""
import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.evaluation import find_forms, fmt, write_csv  # noqa: E402
from formsense.ingest.loader import load_pages  # noqa: E402
from formsense.scoring import score_field, score_issues, summarise, summarise_issues  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402

DEMO_DIR = BACKEND / "data" / "templates" / "demo" / "samples"
DEMO_TPL = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
REAL_TPL = BACKEND / "data" / "templates" / "admission_form.json"
NAMES = {"baseline": "Baseline (OCR + regex)", "vlm": "VLM (Groq)", "formsense": "FormSense"}


def fake_vlm_client(truth: dict, seed: str):
    """A stand-in for the Groq client that 'reads' the form from its ground truth.

    Adds a few deterministic mistakes so the table is not trivially perfect.
    DRY RUN ONLY: it exists to test this script end to end without an API key.
    """
    rng = random.Random(seed)
    fields = {}
    for name, t in truth.items():
        if "checked" in t:
            fields[name] = {"checked": t["checked"] if rng.random() > 0.05 else not t["checked"]}
        elif name == "signature":
            fields[name] = {"present": t["status"] != "missing"}
        elif t["status"] == "missing":
            fields[name] = {"text": None, "legible": True}
        else:
            text = t.get("text", "")
            if rng.random() < 0.12 and len(text) > 2:
                i = rng.randrange(len(text))
                text = text[:i] + text[i + 1:]
            fields[name] = {"text": text, "legible": t["status"] != "unreadable"}
    reply = json.dumps({"fields": fields})
    create = lambda **kw: SimpleNamespace(  # noqa: E731
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        usage=SimpleNamespace(total_tokens=0))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="folder of forms (default: data/synthetic/test)")
    ap.add_argument("--template", help="template JSON (default: admission form, or demo with --demo)")
    ap.add_argument("--demo", action="store_true", help="use the demo form samples (fields only)")
    ap.add_argument("--limit", type=int, help="only the first N forms")
    ap.add_argument("--systems", default="baseline,vlm,formsense", help="comma separated: baseline,vlm,formsense")
    ap.add_argument("--model", help="Groq vision model id (default: GROQ_VLM_MODEL or Qwen 3.8 27B)")
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between VLM calls (rate limit)")
    ap.add_argument("--no-trocr", action="store_true", help="PaddleOCR only (faster)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the VLM answer cache")
    ap.add_argument("--dry-run", action="store_true", help="fake VLM from ground truth; tests the script only")
    ap.add_argument("--out", help="output folder (default: outputs/comparison_vlm)")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    bad = [s for s in systems if s not in NAMES]
    if bad or not systems:
        sys.exit(f"Unknown system(s) {bad}. Choose from: {', '.join(NAMES)}")
    folder = Path(args.dir) if args.dir else (DEMO_DIR if args.demo else BACKEND / "data" / "synthetic" / "test")
    tpl_path = args.template or (DEMO_TPL if args.demo else REAL_TPL)
    template, blank = load_template(tpl_path)
    kinds = {f.name: f.type for f in template.fields}
    forms = find_forms(folder)
    forms = forms[:args.limit] if args.limit else forms
    if not forms:
        sys.exit(f"No forms with ground truth found in {folder}.\n"
                 "Generate some: python -m formsense.generator.generate --count 40 --split test "
                 "--seed 100000 --out data/synthetic/test")
    with_issues = all(lab is not None for _, _, lab in forms)
    if with_issues:
        from formsense.rules.audit import audit

    out_dir = Path(args.out) if args.out else BACKEND / "outputs" / ("comparison_vlm_dryrun" if args.dry_run
                                                                     else "comparison_vlm")
    if args.dry_run:
        print("*** DRY RUN: the VLM is simulated from the ground truth. These numbers are not results. ***\n")

    ocr = baseline = vlm = None
    if "formsense" in systems or "baseline" in systems:
        from formsense.vision.ocr import FieldOCR
        ocr = FieldOCR(use_trocr=not args.no_trocr)
    if "baseline" in systems:
        from formsense.baseline import BaselineAuditor
        baseline = BaselineAuditor(template, reader=ocr.paddle)
    if "vlm" in systems:
        from formsense.vlm import VLMAuditor
        vlm = VLMAuditor(template, model=args.model, cache_dir=None if (args.no_cache or args.dry_run) else
                         BACKEND / "outputs" / "vlm_cache",
                         client=None)
        if not args.dry_run:
            vlm.client   # fail now, not after the OCR models loaded, if the API key is missing

    per_form = []          # one dict per form: {system: fields} or errors
    secs = {s: 0.0 for s in systems}
    vlm_calls = vlm_tokens = vlm_cached = 0
    failures = []
    for i, (img_path, truth, label) in enumerate(forms, 1):
        page = load_pages(img_path)[0]
        got, err = {}, None
        for s in systems:
            t0 = time.time()
            try:
                if s == "baseline":
                    got[s] = baseline.run(page).fields
                elif s == "formsense":
                    got[s] = analyze_page(page, template, blank, ocr=ocr).fields
                else:
                    if args.dry_run:
                        vlm._client = fake_vlm_client(truth, img_path.stem)
                    out = vlm.run(page)
                    got[s] = out.fields
                    vlm_tokens += out.tokens
                    vlm_cached += out.from_cache
                    vlm_calls += 1
                    if not args.dry_run and not out.from_cache and args.delay > 0 and i < len(forms):
                        time.sleep(args.delay)
            except Exception as e:  # one bad form must not stop a long evaluation
                err = f"{s}: {e}"
                failures.append({"form": img_path.stem, "system": s, "error": str(e)})
                print(f"  ! {img_path.name}: {err}")
                break
            secs[s] += time.time() - t0
        per_form.append({"path": img_path, "truth": truth, "label": label, "got": got, "ok": err is None})
        print(f"[{i}/{len(forms)}] {img_path.name}: " + ("failed" if err else "done"))

    good = [f for f in per_form if f["ok"]]
    if not good:
        sys.exit("Every form failed; nothing to score. See the errors above.")

    rows = {s: [] for s in systems}
    issue_rows = {s: [] for s in systems}
    detail, issue_detail = [], []
    for f in good:
        for s in systems:
            fields = f["got"][s]
            pred = {r.name: r for r in fields}
            for name, tr in f["truth"].items():
                r = score_field(kinds[name], pred[name], tr)
                rows[s].append(r)
                p = pred[name]
                detail.append({"form": f["path"].stem, "field": name, "system": s, "correct": r["correct"],
                               "truth": tr.get("text", tr.get("checked", tr["status"])),
                               "truth_status": tr["status"],
                               "predicted": p.raw_text or p.value or p.status.value,
                               "predicted_status": p.status.value})
            if with_issues:
                today = dt.date.fromisoformat(f["label"]["reference_date"])
                res = score_issues(audit(fields, today=today).issues, f["label"]["issues"])
                res["truth_n"] = len(f["label"]["issues"])
                issue_rows[s].append(res)
                issue_detail.append({"form": f["path"].stem, "system": s, "true_issues": res["truth_n"],
                                     "found": res["tp"], "missed": " | ".join(res["missed"]),
                                     "false_alarms": " | ".join(res["extra"])})

    summary = {s: summarise(rows[s]) for s in systems}
    if with_issues:
        for s in systems:
            summary[s].update(summarise_issues(issue_rows[s]))
    for s in systems:
        summary[s]["Seconds per form"] = round(secs[s] / len(forms), 1)
    if vlm is not None and vlm_calls:
        summary["vlm"]["Tokens per form (new calls)"] = round(vlm_tokens / max(1, vlm_calls - vlm_cached))\
            if vlm_calls > vlm_cached else None

    n_issues = sum(len(f["label"]["issues"]) for f in good) if with_issues else 0
    head = f"# Baseline vs. VLM vs. FormSense ({len(good)} forms, {len(rows[systems[0]])} fields"
    head += f", {n_issues} true issues)" if with_issues else ")"
    if args.dry_run:
        head = "# DRY RUN (simulated VLM, not results)\n\n" + head
    lines = [head, "", "| Metric | " + " | ".join(NAMES[s] for s in systems) + " |",
             "| --- | " + " | ".join("---" for _ in systems) + " |"]
    metrics = list(dict.fromkeys(m for s in systems for m in summary[s]))
    lines += [f"| {m} | " + " | ".join(fmt(summary[s].get(m)) for s in systems) + " |" for m in metrics]
    md = "\n".join(lines) + "\n"
    if with_issues:
        md += ("\nIssues are scored end to end: each system's fields go through the same rules "
               "(rules/audit.py); an issue counts when the same rule fires on the same fields.\n")
    if failures:
        md += f"\n{len(failures)} failure(s), excluded from the table:\n" + \
              "\n".join(f"- {x['form']} ({x['system']}): {x['error']}" for x in failures) + "\n"
    if vlm is not None:
        md += (f"\nVLM model: `{vlm.model}`. {vlm_calls} answered ({vlm_cached} from cache). "
               "The VLM gives no confidence score and no pixel evidence; ticks and signatures are its "
               "own judgement of the image.\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "fields.csv", detail)
    write_csv(out_dir / "issues.csv", issue_detail)
    print("\n" + md)
    print(f"Saved {out_dir.relative_to(BACKEND).as_posix() if out_dir.is_relative_to(BACKEND) else out_dir}/ "
          "summary.md, summary.json, fields.csv" + (", issues.csv" if with_issues else ""))


if __name__ == "__main__":
    main()
