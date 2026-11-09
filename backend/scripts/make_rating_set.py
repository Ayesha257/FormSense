"""Build the explanation rating set: LLM alone vs. LLM + RAG, blinded, ready for two raters.

    # real run: FormSense on each form, then an LLM explanation with and without guidelines
    python scripts/make_rating_set.py --dir data/synthetic/test --limit 15

    # only forms that have at least one issue are worth rating (default --min-issues 1)
    # 15 forms x 2 conditions = the roadmap's ~30 outputs

    # try the whole flow without OCR models or an API key (templated text from the ground
    # truth; the app shows a TEST SET banner. Never use this for real ratings)
    python scripts/make_rating_set.py --dir data/synthetic/dev --limit 6 --offline

Writes to eval/rating/:
  rating_set.json   shuffled items, no condition shown to raters
  rating_key.json   item id -> llm_only | llm_rag (do not share with the raters)
  images/           annotated form for each item

Explainer: Easha's formsense.llm explainer through formsense/explain.py. "LLM alone" gets the
issues only; "LLM + RAG" gets the same prompt plus the retrieved guideline passages, so the two
arms differ in nothing else. A real Groq key is required (the script refuses to rate templated
fallback text as if an LLM had written it). Personal data (CNIC, phone, email) is masked by the
explainer before any text reaches the model. Use --explainer module:function to plug in another one.
"""
import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense import rating as R  # noqa: E402
from formsense.evaluation import find_forms  # noqa: E402
from formsense.schema import AuditReport, GuidelineCitation, Issue  # noqa: E402
from formsense.summary import png_bytes  # noqa: E402

def llm_explainer(report: AuditReport) -> str:
    """Easha's explainer. report.citations empty = "LLM alone", filled = "LLM + RAG"."""
    from formsense.explain import LLMModelUnavailable, explain_report, explanation_text
    try:
        return explanation_text(explain_report(report, use_rag=True, offline=False))
    except LLMModelUnavailable as e:
        raise SystemExit(str(e))
    except Exception as e:
        raise SystemExit(f"The explainer failed: {e}")


def offline_explainer(report: AuditReport) -> str:
    """Templated text for trying the app. NOT a real explanation."""
    lines = ["Here is what needs fixing on your form:"]
    lines += [f"- {i.message}" for i in report.issues] or ["- Nothing: the form looks fine."]
    if report.citations:
        lines.append("Guideline: " + " ".join(f"{c.source}: {c.text[:140]}..." for c in report.citations[:2]))
    return "\n".join(lines)


def pick_explainer(spec: str, offline: bool):
    if offline:
        return offline_explainer
    if spec == "auto":
        from formsense.explain import is_offline
        if is_offline():
            raise SystemExit("No GROQ_API_KEY in backend/.env. Real ratings need real LLM explanations "
                             "(the offline template would be rated as if an LLM wrote it). "
                             "Add the key, or use --offline just to try the tool.")
        return llm_explainer
    module, _, attr = spec.partition(":")
    import importlib
    return getattr(importlib.import_module(module), attr)


# ---------------- per form ----------------

def facts_for(fields, labels) -> list[str]:
    out = []
    for f in fields:
        who = labels.get(f.name, f.name)
        if f.checked is not None:
            if f.checked:
                out.append(f"{who}: ticked")
        elif f.status.value == "missing":
            out.append(f"{who}: empty")
        else:
            out.append(f"{who}: {f.raw_text or f.value or 'present'}" +
                       ("" if f.status.value == "ok" else f" ({f.status.value})"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(BACKEND / "data" / "synthetic" / "test"))
    ap.add_argument("--limit", type=int, default=15, help="forms to use (each gives 2 items)")
    ap.add_argument("--min-issues", type=int, default=1, help="skip forms with fewer issues")
    ap.add_argument("--seed", type=int, default=7, help="shuffle seed")
    ap.add_argument("--out", default=str(R.RATING_DIR))
    ap.add_argument("--explainer", default="auto", help="'auto', or module:function")
    ap.add_argument("--offline", action="store_true", help="templated text from ground truth, no models")
    ap.add_argument("--no-trocr", action="store_true")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between LLM calls")
    args = ap.parse_args()

    out = Path(args.out)
    forms = find_forms(Path(args.dir))
    if args.offline:
        forms = [f for f in forms if f[2] is not None]
    if not forms:
        sys.exit(f"No forms found in {args.dir}. Generate some with formsense.generator.generate.")
    explain = pick_explainer(args.explainer, args.offline)

    from formsense.rag import GuidelineRetriever
    from formsense.rag.retriever import field_labels
    from formsense.vision.template import DEFAULT_TEMPLATE
    labels = field_labels(DEFAULT_TEMPLATE)
    if args.offline:    # keyword search + fake embeddings: no model download
        import tempfile

        from langchain_core.embeddings import DeterministicFakeEmbedding
        # a throwaway index: fake vectors must never overwrite the real guidelines index
        retriever = GuidelineRetriever(index_dir=Path(tempfile.mkdtemp(prefix="formsense_offline_index_")),
                                       embeddings=DeterministicFakeEmbedding(size=32), mode="tfidf").ensure()
    else:
        retriever = GuidelineRetriever().ensure()
        from formsense.pipeline import FormSense
        fs = FormSense(use_trocr=not args.no_trocr, retriever=retriever)

    from formsense.schema import FieldResult
    from formsense.vision.draw import annotate
    items, key = [], {}
    used = 0
    (out / "images").mkdir(parents=True, exist_ok=True)
    for img_path, truth, label in forms:
        if used >= args.limit:
            break
        if args.offline:
            fields = [FieldResult(**f) for f in label["fields"]]
            issues = [Issue(**i) for i in label["issues"]]
            aligned = None
        else:
            from formsense.ingest.loader import load_pages
            today = dt.date.fromisoformat(label["reference_date"]) if label else dt.date.today()
            res = fs.run(load_pages(img_path)[0], today=today, form_id=img_path.stem)
            fields, issues, aligned = res.fields, res.issues, res.vision.aligned
        if len(issues) < args.min_issues:
            continue
        cites = []
        for i in issues:
            c = retriever.for_issue(i, labels)
            if c:
                cites.append(c[0])
        form_id = img_path.stem
        image = None
        if aligned is not None:
            image = f"images/{form_id}.png"
            (out / image).write_bytes(png_bytes(annotate(aligned, fields)))
        issue_rows = [{"id": i.id, "type": i.type.value, "severity": i.severity.value,
                       "fields": i.fields, "message": i.message} for i in issues]
        for cond, use_cites in (("llm_only", []), ("llm_rag", cites)):
            report = AuditReport(form_id=form_id, fields=fields, issues=issues, citations=use_cites)
            out_ = explain(report)
            text = out_ if isinstance(out_, str) else getattr(out_, "explanation", None)
            if not text:
                sys.exit(f"The explainer returned nothing for {form_id} ({cond}).")
            items.append({"id": "", "form_id": form_id, "explanation": text, "issues": issue_rows,
                          "facts": facts_for(fields, labels), "image": image, "_cond": cond})
            if not args.offline and args.delay > 0:
                time.sleep(args.delay)
        used += 1
        print(f"[{used}/{args.limit}] {form_id}: {len(issues)} issue(s)")

    if not items:
        sys.exit("No forms with issues to rate. Lower --min-issues or generate forms with injected errors.")
    random.Random(args.seed).shuffle(items)       # blind: pairs are not next to each other
    for n, it in enumerate(items, 1):
        it["id"] = f"item_{n:03d}"
        key[it["id"]] = it.pop("_cond")
    R.save_set(out / "rating_set.json", items, key, out / "rating_key.json")
    if args.offline:
        doc = json.loads((out / "rating_set.json").read_text(encoding="utf-8"))
        doc["offline_test_set"] = True
        (out / "rating_set.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(items)} items ({used} forms x 2 conditions) to {out}.\n"
          "Next: python -m streamlit run app/rate_explanations.py   (each rater, on their own copy)")


if __name__ == "__main__":
    main()
