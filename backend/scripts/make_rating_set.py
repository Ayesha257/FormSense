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

Explainer: uses formsense.llm.explain(report) when the language side provides it
(contract: explain(AuditReport) -> str | AuditReport; report.citations empty means
"LLM alone"). Otherwise a small built-in Groq explainer is used. Personal data
(CNIC, phone, email) is masked before any text reaches the LLM.
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense import rating as R  # noqa: E402
from formsense.evaluation import find_forms  # noqa: E402
from formsense.schema import AuditReport, GuidelineCitation, Issue  # noqa: E402
from formsense.summary import explanation_text, find_llm_explainer, plain_summary, png_bytes  # noqa: E402

PII = [(re.compile(r"\b\d{5}-?\d{7}-?\d\b"), "[CNIC]"),
       (re.compile(r"\b03\d{2}-?\d{7}\b"), "[PHONE]"),
       (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]")]


def mask_pii(text: str) -> str:
    for rx, token in PII:
        text = rx.sub(token, text)
    return text


# ---------------- explainers ----------------

def groq_explainer(report: AuditReport) -> str:
    """Built-in fallback. Citations present = LLM + RAG, absent = LLM alone."""
    from formsense import groq_util
    try:
        client = groq_util.make_client()
    except groq_util.GroqError as e:
        raise SystemExit(f"{e}\nUse --offline to try the flow without a key.")
    facts = "\n".join(f"- [{i.severity.value}] {mask_pii(i.message)}" for i in report.issues) or "- no problems found"
    prompt = ("An applicant uploaded a filled university admission form. An automatic checker found these "
              f"problems:\n{facts}\n\n")
    if report.citations:
        prompt += "Relevant passages from the university's form-filling guidelines:\n" + "\n".join(
            f"- ({c.source}) {c.text}" for c in report.citations) + "\n\n"
    prompt += ("Write a short, friendly explanation for the applicant in plain English: say what is wrong, "
               "why it matters and exactly how to fix it. Use ONLY the facts above"
               + (" and the guideline passages" if report.citations else "")
               + ". Do not invent problems, values or rules. No headings; at most 150 words.")
    try:
        r = groq_util.chat(client, env_var="GROQ_TEXT_MODEL", default_model=groq_util.DEFAULT_TEXT_MODEL,
                           reasoning_effort=os.getenv("GROQ_TEXT_REASONING", "none") or None,
                           model=groq_util.text_model(), temperature=0.2, max_completion_tokens=1500,
                           messages=[{"role": "user", "content": prompt}])
    except groq_util.GroqError as e:
        raise SystemExit(str(e))
    text = (r.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
        return find_llm_explainer() or groq_explainer
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
            text = explanation_text(explain(report))
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
