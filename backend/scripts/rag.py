"""Guidelines retrieval tool.

    python scripts/rag.py build                         # (re)build the index
    python scripts/rag.py ask "Can my mother sign the form?"
    python scripts/rag.py ask "two programs ticked" --mode dense --k 3
    python scripts/rag.py eval                          # compare dense / tfidf / hybrid

    python scripts/rag.py eval --issues-from data/synthetic/dev --save   # refresh rule issues

The index also rebuilds itself automatically whenever a file in
data/guidelines/ changes. `eval` uses data/guidelines/eval_queries.json:
applicant questions plus issues produced by the rules, each with the guideline
section(s) that should come back first. Update the questions and the section
names in this script (FIELD_SECTION, RULE_SECTION) when the guidelines change.
"""
import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.rag.retriever import (GUIDELINES_DIR, HEADING_BOOST, MODES, GuidelineRetriever,  # noqa: E402
                                     field_labels)
from formsense.schema import Issue, IssueType, Severity  # noqa: E402

EVAL_FILE = GUIDELINES_DIR / "eval_queries.json"
OUT = BACKEND / "outputs" / "rag_eval"


def cmd_build(r: GuidelineRetriever, _):
    t = time.time()
    n = r.build()
    print(f"Indexed {n} chunks from {len(list(GUIDELINES_DIR.glob('*.md')))} file(s) "
          f"in {time.time() - t:.1f}s -> {r.index_dir.relative_to(BACKEND).as_posix()}")


def cmd_ask(r: GuidelineRetriever, args):
    for i, p in enumerate(r.search(args.question, k=args.k, mode=args.mode,
                                   expand=not args.no_synonyms), 1):
        print(f"\n[{i}] score {p.score:.3f}  {p.ref}\n" + "-" * 70)
        print(p.text)


# Which guideline section explains each rule (used to build the issue test set).
FIELD_SECTION = {
    "full_name": "Full Name", "father_name": "Father's Name", "cnic": "CNIC Number",
    "date_of_birth": "Date of Birth", "email": "Email Address", "phone": "Mobile Number",
    "address": "Postal Address", "matric_year": "Examination Years", "inter_year": "Examination Years",
    "matric_total": "Total and Obtained Marks", "matric_obtained": "Total and Obtained Marks",
    "inter_total": "Total and Obtained Marks", "inter_obtained": "Total and Obtained Marks",
    "matric_percentage": "Percentage", "inter_percentage": "Percentage",
    "declaration_name": "Declarant Name", "declaration_date": "Declaration Date",
    "signature": "Signature of Applicant", "gender_male": "Gender", "gender_female": "Gender",
    "program_bscs": "Program Applied For", "program_bsse": "Program Applied For",
    "program_bsai": "Program Applied For",
}
RULE_SECTION = {
    "invalid_cnic": ["CNIC Number"], "invalid_email": ["Email Address"], "invalid_phone": ["Mobile Number"],
    "underage": ["Date of Birth"], "declaration_date_future": ["Declaration Date"],
    "declaration_before_dob": ["Declaration Date", "Date of Birth"],
    "cnic_gender_parity": ["CNIC Number", "Gender"], "year_order": ["Examination Years"],
    "marks_exceed_total": ["Total and Obtained Marks"], "percentage_mismatch": ["Percentage"],
    "name_mismatch": ["Declarant Name"],
}
ALSO = {"unreadable_field": ["Writing the Form"], "placeholder_value": ["Values That Are Not Accepted"],
        "correction_mark": ["Corrections"]}


def expected_sections(issue: dict) -> list[str]:
    if issue["id"] in RULE_SECTION:
        return RULE_SECTION[issue["id"]]
    own = list(dict.fromkeys(FIELD_SECTION.get(f, f) for f in issue["fields"]))
    return own + ALSO.get(issue["id"], [])


def issues_from_labels(folder: Path) -> list[dict]:
    """One test case per distinct (rule, fields) found in generator label files."""
    seen = {}
    for lab in sorted(Path(folder, "labels").glob("*.json")):
        for i in json.loads(lab.read_text(encoding="utf-8"))["issues"]:
            seen.setdefault((i["id"], tuple(i["fields"])), i)
    return [{"id": i["id"], "type": i["type"], "fields": i["fields"], "message": i["message"],
             "sections": expected_sections(i)} for i in seen.values()]


def _score(r, queries, mode, expand, boost=0.0, labels=None):
    hit1 = hit3 = rr = 0.0
    misses = []
    for q in queries:
        if "issue" in q:
            hits = r.search_for_issue(q["issue"], labels, k=5, mode=mode, expand=expand, heading_boost=boost)
        else:
            hits = r.search(q["query"], k=5, mode=mode, expand=expand)
        sections = [p.section for p in hits]
        want = [w.lower() for w in q["sections"]]
        rank = next((i for i, s in enumerate(sections, 1) if any(w in s.lower() for w in want)), None)
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        if rank != 1:
            misses.append({"query": q["query"], "want": q["sections"][0], "got": sections[0], "rank": rank})
    n = len(queries)
    return {"Hit@1 (%)": round(100 * hit1 / n, 1), "Hit@3 (%)": round(100 * hit3 / n, 1),
            "MRR": round(rr / n, 3), "misses": misses}


def cmd_eval(r: GuidelineRetriever, args):
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if args.issues_from:
        data["issues"] = issues_from_labels(Path(args.issues_from))
        if args.save:
            EVAL_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
            print(f"Saved {len(data['issues'])} issue cases to {EVAL_FILE.relative_to(BACKEND).as_posix()}")
    template = field_labels()
    sets = {
        "Applicant questions": data["questions"],
        "Issues from the rules step": [
            {"issue": Issue(id=x.get("id", f"q{i}"), type=IssueType(x["type"]), severity=Severity.ERROR,
                            fields=x["fields"], message=x["message"]),
             "query": x["message"], "sections": x["sections"]} for i, x in enumerate(data["issues"])],
    }
    names = {"tfidf": "Keyword (TF-IDF)", "dense": "Semantic (MiniLM)", "hybrid": "Hybrid (0.7 sem + 0.3 kw)"}
    results, lines = {}, []
    for set_name, queries in sets.items():
        lines += [f"### {set_name} ({len(queries)})", "",
                  "| Method | Synonyms | Hit@1 (%) | Hit@3 (%) | MRR |", "| --- | --- | --- | --- | --- |"]
        is_issues = "issue" in queries[0]
        settings = [(m, e, 0.0) for m in ("tfidf", "dense", "hybrid") for e in (False, True)]
        if is_issues:
            settings.append(("hybrid", True, HEADING_BOOST))
        for mode, expand, boost in settings:
            m = _score(r, queries, mode, expand, boost, template)
            tag = " + heading boost" if boost else ""
            results[f"{set_name} | {mode} | synonyms={expand} | boost={boost}"] = m
            lines.append(f"| {names[mode]}{tag} | {'yes' if expand else 'no'} | {m['Hit@1 (%)']} | "
                         f"{m['Hit@3 (%)']} | {m['MRR']} |")
        lines.append("")
    md = "# Guideline retrieval evaluation\n\n" + "\n".join(lines)
    print(md)
    defaults = (f"Applicant questions | {r.mode} | synonyms={r.expand} | boost=0.0",
                f"Issues from the rules step | {r.mode} | synonyms={r.expand} | boost={HEADING_BOOST}")
    for key, m in results.items():
        if key in defaults and m["misses"]:
            print(f"Default setting misses ({key.split(' |')[0]}):")
            for x in m["misses"]:
                print(f"   {x['query'][:70]!r}\n      want {x['want']!r}, got {x['got']!r} (rank {x['rank'] or '>5'})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT.relative_to(BACKEND).as_posix()}/summary.md and results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=2)
    a.add_argument("--mode", choices=MODES, default="hybrid")
    a.add_argument("--no-synonyms", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--issues-from", help="generator folder (with labels/) to take the rule issues from")
    e.add_argument("--save", action="store_true", help="with --issues-from: store them in eval_queries.json")
    args = ap.parse_args()
    r = GuidelineRetriever()
    {"build": cmd_build, "ask": cmd_ask, "eval": cmd_eval}[args.cmd](r, args)


if __name__ == "__main__":
    main()
