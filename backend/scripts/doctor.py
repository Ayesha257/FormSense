"""One command that says what works and what does not: setup, guidelines, RAG, explainer, Groq, OCR.

    python scripts/doctor.py               # offline checks (no network, no OCR models)
    python scripts/doctor.py --online      # also test the Groq text and vision models (needs GROQ_API_KEY)
    python scripts/doctor.py --ocr         # also load PaddleOCR and read a test image
    python scripts/doctor.py --online --ocr --fix     # everything; --fix rebuilds a broken guidelines index

Each line is PASS, WARN (works, but look at it) or FAIL (broken, with what to do). Exit code is 1 only if
something FAILED, so it can run before every demo and rehearsal.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

RESULTS = []


def record(status, name, detail=""):
    RESULTS.append((status, name))
    mark = {"PASS": "[ PASS ]", "WARN": "[ WARN ]", "FAIL": "[ FAIL ]"}[status]
    print(f"{mark} {name}" + (f"\n         {detail}" if detail else ""))


def check(name):
    """Run a check; an exception is a FAIL with the error, never a crash of the doctor itself."""
    def deco(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record("FAIL", name, f"{type(e).__name__}: {e}")
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true", help="call Groq (text + vision model)")
    ap.add_argument("--ocr", action="store_true", help="load PaddleOCR and read a test image")
    ap.add_argument("--fix", action="store_true", help="rebuild the guidelines index if it is broken")
    ap.add_argument("--fake-embeddings", action="store_true", help=argparse.SUPPRESS)   # used by the tests
    args = ap.parse_args()

    from formsense import groq_util
    from formsense.rag import GuidelineRetriever
    from formsense.rag.retriever import GUIDELINES_DIR, field_labels, guideline_files

    print("FormSense doctor\n")

    # ---------- environment ----------
    groq_util.load_env()
    has_key = bool(os.getenv("GROQ_API_KEY"))

    @check("Environment: backend/.env")
    def _env():
        if (BACKEND / ".env").exists():
            record("PASS", "Environment: backend/.env found" + (", GROQ_API_KEY set" if has_key else ""))
        else:
            record("WARN", "Environment: no backend/.env",
                   "Copy .env.example to .env and add GROQ_API_KEY. Without it, explanations are plain templates.")
        if has_key:
            record("PASS", f"Groq models: vision={groq_util.vision_model()}, text={groq_util.text_model()}")
            for m in (groq_util.vision_model(), groq_util.text_model()):
                if m in groq_util.RETIRED_MODELS:
                    record("FAIL", f"Groq model '{m}' was retired", "Set GROQ_VLM_MODEL / GROQ_LLM_MODEL in .env.")

    # ---------- guidelines ----------
    @check("Guidelines")
    def _guidelines():
        files = guideline_files(GUIDELINES_DIR)
        if not files:
            record("FAIL", "Guidelines: no .md file in data/guidelines/", "Dania's guidelines document goes there.")
            return
        text = "\n".join(f.read_text(encoding="utf-8") for f in files)
        record("PASS", f"Guidelines: {len(files)} file(s), {len(text.split())} words")
        if "SAMPLE / DRAFT" in text:
            record("WARN", "Guidelines are still the SAMPLE / DRAFT text",
                   "Replace data/guidelines/*.md with the final document (the index rebuilds by itself), then update "
                   "data/guidelines/eval_queries.json and run: python scripts/rag.py eval")

    # ---------- RAG ----------
    if args.fake_embeddings:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        retriever = GuidelineRetriever(index_dir=Path(tempfile.mkdtemp(prefix="doctor_index_")),
                                       embeddings=DeterministicFakeEmbedding(size=32), mode="tfidf")
    else:
        retriever = GuidelineRetriever()

    @check("RAG index")
    def _index():
        fresh = retriever.is_fresh()
        manifest = retriever.index_dir / "manifest.json"
        if not fresh:
            why = "no index yet" if not manifest.exists() else "index was built from other text or another embedding model"
            if args.fix or not manifest.exists():
                n = retriever.build()
                record("PASS", f"RAG index: {why}; rebuilt ({n} chunks)")
            else:
                record("WARN", f"RAG index is stale ({why})", "It rebuilds on first use; or run with --fix.")
                return
        retriever.ensure()
        dim = int(retriever._store.index.d)
        q = len(retriever.embeddings.embed_query("test"))
        if dim != q:
            record("FAIL", f"RAG index has {dim}-dimension vectors, the embedding model gives {q}",
                   "Delete data/guidelines/.index (or run with --fix); it rebuilds on the next run.")
        else:
            record("PASS", f"RAG index: {retriever._store.index.ntotal} chunks, {dim}-dimension vectors match the model")

    @check("RAG retrieval quality")
    def _quality():
        retriever.ensure()
        sys.path.insert(0, str(BACKEND / "scripts"))
        import rag as rag_cli
        queries = json.loads((GUIDELINES_DIR / "eval_queries.json").read_text(encoding="utf-8"))
        labels = field_labels()
        mode = retriever.mode
        from formsense.rag.retriever import HEADING_BOOST
        from formsense.schema import Issue, IssueType, Severity
        issue_cases = [{"issue": Issue(id=x.get("id", f"q{i}"), type=IssueType(x["type"]), severity=Severity.ERROR,
                                       fields=x["fields"], message=x["message"]),
                        "query": x["message"], "sections": x["sections"]}
                       for i, x in enumerate(queries.get("issues", []))]
        for set_name, items, boost in (("applicant questions", queries.get("questions", []), 0.0),
                                       ("issues from the rules", issue_cases, HEADING_BOOST)):
            if not items:
                continue
            m = rag_cli._score(retriever, items, mode, retriever.expand, boost, labels)
            h1, h3 = m["Hit@1 (%)"], m["Hit@3 (%)"]
            status = "PASS" if h3 >= 90 else "WARN" if h3 >= 75 else "FAIL"
            record(status, f"RAG retrieval, {set_name}: right section first {h1}% of the time, in top 3 {h3}% (n={len(items)})",
                   "" if status == "PASS" else "Check data/guidelines/eval_queries.json matches the current guidelines "
                                               "section names, then: python scripts/rag.py eval")
            for miss in m["misses"][:3]:
                if status != "PASS":
                    print(f"           miss: {miss['query'][:70]!r} wanted {miss['want']!r}, got {miss['got']!r}")

    # ---------- explainer wiring (RAG -> LLM) ----------
    @check("RAG -> explainer")
    def _wiring():
        from formsense.explain import explain_report
        from formsense.llm.explanation_schema import ExplanationOutput, IssueExplanation
        from formsense.schema import AuditReport, FieldResult, FieldStatus, Issue, IssueType, Severity
        retriever.ensure()
        issue = Issue(id="no_selection", type=IssueType.SELECTION, severity=Severity.ERROR,
                      fields=["program_bscs", "program_bsse", "program_bsai"], message="No program is ticked")
        cites = retriever.for_issue(issue, field_labels())
        if not cites:
            record("FAIL", "RAG returned no guideline for a 'no program ticked' issue"); return

        class Spy:
            prompt = ""

            def explain(self, system_prompt, user_prompt):
                Spy.prompt = user_prompt
                return ExplanationOutput(summary="s", issues=[IssueExplanation(issue_id=issue.id, explanation="e")],
                                         overall_recommendation=None)
        rep = AuditReport(form_id="doctor", fields=[FieldResult(name="program_bscs", bbox=(0, 0, 1, 1),
                                                                status=FieldStatus.OK, checked=False)],
                          issues=[issue], citations=[cites[0]])
        explain_report(rep, client=Spy(), use_rag=True)
        ok = cites[0].text[:40] in Spy.prompt
        record("PASS" if ok else "FAIL", "RAG -> explainer: the retrieved guideline reaches the LLM prompt" if ok else
               "The guideline text is missing from the LLM prompt (formsense/explain.py)")

    # ---------- Groq ----------
    if args.online:
        @check("Groq")
        def _groq():
            if not has_key:
                record("FAIL", "Groq: GROQ_API_KEY is not set"); return
            client = groq_util.make_client()
            r = groq_util.chat(client, env_var="GROQ_LLM_MODEL", default_model=groq_util.DEFAULT_TEXT_MODEL,
                               reasoning_effort="none", model=groq_util.text_model(), temperature=0,
                               max_completion_tokens=50, messages=[{"role": "user", "content": "Reply with the word OK."}])
            record("PASS", f"Groq text model {groq_util.text_model()} answered: {(r.choices[0].message.content or '').strip()[:30]!r}")

        @check("Groq vision")
        def _vision():
            if not has_key:
                return
            import cv2
            import numpy as np
            from formsense.vlm.groq_vlm import encode_image
            img = np.full((140, 520, 3), 255, np.uint8)
            cv2.putText(img, "FormSense 2026", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
            b64, _ = encode_image(img)
            r = groq_util.chat(groq_util.make_client(), reasoning_effort="none", model=groq_util.vision_model(),
                               temperature=0.1, max_completion_tokens=100,
                               messages=[{"role": "user", "content": [
                                   {"type": "text", "text": "Read the text in the image. Reply with the text only."},
                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
            text = (r.choices[0].message.content or "").strip()
            record("PASS" if "formsense" in text.lower() else "WARN",
                   f"Groq vision model {groq_util.vision_model()} read: {text[:40]!r}")

    # ---------- OCR ----------
    if args.ocr:
        @check("OCR")
        def _ocr():
            import cv2
            import numpy as np
            from formsense.vision.ocr import FieldOCR
            img = np.full((120, 480, 3), 255, np.uint8)
            cv2.putText(img, "Ayesha Khan", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
            res = FieldOCR(use_trocr=False).read(img)
            ok = "ayesha" in res.text.lower().replace(" ", "")
            record("PASS" if ok else "WARN", f"PaddleOCR read {res.text!r} (confidence {res.confidence:.2f})")

    # ---------- summary ----------
    n = {s: sum(1 for r, _ in RESULTS if r == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} warning(s), {n['FAIL']} failed")
    if n["FAIL"]:
        print("Fix the FAIL lines above, then run the doctor again.")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
