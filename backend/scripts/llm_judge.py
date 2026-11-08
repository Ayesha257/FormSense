"""LLM-as-judge for the explanation rating set (LLM alone vs LLM + RAG).

Two judge models score every blinded item 1-5 on faithfulness, clarity and usefulness.
The judges only see the form facts, the issues and the explanation, never the condition.
Results are LLM-judge ratings, NOT human ratings: report them as such.

    python scripts/llm_judge.py
    python scripts/llm_judge.py --dir eval/rating --delay 2

Judge models (set in .env, or edit the defaults below):
    GROQ_JUDGE_A=openai/gpt-oss-120b
    GROQ_JUDGE_B=qwen/qwen3.8-27b
Check which models your key can use: python scripts/groq_models.py
Use synthetic forms only: facts are sent to a third-party API.

Writes into --dir: llm_judge_cache.json (resumable), llm_judge_items.csv,
llm_judge_results.json, llm_judge_results.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

CRITERIA = ("faithfulness", "clarity", "usefulness")

SYSTEM = """You are a strict evaluator of explanations written for people whose university admission form was checked automatically.
You are given the facts read from the form, the list of issues found, and one explanation to judge.
Score the explanation from 1 to 5 on each criterion:
- faithfulness: every claim is supported by the facts and issues; nothing invented; no issue omitted. 5 = fully faithful, 1 = mostly wrong or invented.
- clarity: plain English a student understands; well organised; not vague. 5 = very clear, 1 = confusing.
- usefulness: tells the person exactly what to fix and how, so they can act on it. 5 = precise and actionable, 1 = no practical guidance.
Use the whole scale. Do not reward length. Reply with ONLY one JSON object:
{"faithfulness": <1-5>, "clarity": <1-5>, "usefulness": <1-5>, "reason": "<one short sentence>"}"""


def build_prompt(item: dict) -> str:
    issues = "\n".join(f"- [{i['severity']}] {i['message']}" for i in item["issues"]) or "(none)"
    facts = "\n".join(f"- {f}" for f in item["facts"])
    return (f"FORM FACTS:\n{facts}\n\nISSUES FOUND:\n{issues}\n\n"
            f"EXPLANATION TO JUDGE:\n{item['explanation']}")


def make_llm(model: str):
    from langchain_groq import ChatGroq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (put it in backend/.env)")
    # generous max_tokens: gpt-oss models spend part of it on reasoning
    return ChatGroq(model=model, api_key=key, temperature=0, max_tokens=3000)


def judge_once(llm, item: dict, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            text = llm.invoke([("system", SYSTEM), ("human", build_prompt(item))]).content
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            data = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
            out = {c: min(5, max(1, int(round(float(data[c]))))) for c in CRITERIA}
            out["reason"] = str(data.get("reason", ""))[:300]
            return out
        except Exception as e:  # rate limit, malformed JSON, network
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"judge failed for {item['id']}: {last!r}")


def load_key(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("items", raw) if isinstance(raw, dict) else raw
    if isinstance(raw, list):
        raw = {r["id"]: r for r in raw}
    out = {}
    for k, v in raw.items():
        out[k] = v.get("condition", v.get("cond")) if isinstance(v, dict) else v
    return out


def kappa(a, b, weighted=False, k=5):
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x - 1][y - 1] += 1
    ra = [sum(r) for r in O]
    cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2 / (k - 1) ** 2) if weighted else (0 if i == j else 1)
            num += w * O[i][j] / n
            den += w * ra[i] * cb[j] / (n * n)
    return 1 - num / den if den else float("nan")


def landis_koch(x):
    if x != x:
        return "undefined (no variation in ratings)"
    for lim, name in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                      (0.8, "substantial"), (1.01, "almost perfect")]:
        if x < lim:
            return name
    return "almost perfect"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="eval/rating")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--judge-a", default=None)
    ap.add_argument("--judge-b", default=None)
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    d = Path(args.dir)
    items = json.loads((d / "rating_set.json").read_text(encoding="utf-8"))["items"]
    key = load_key(d / "rating_key.json")
    judges = {
        "judge_A": args.judge_a or os.environ.get("GROQ_JUDGE_A", "openai/gpt-oss-120b"),
        "judge_B": args.judge_b or os.environ.get("GROQ_JUDGE_B", "qwen/qwen3.8-27b"),
    }
    cache_path = d / "llm_judge_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    for jname, model in judges.items():
        llm = make_llm(model)
        for n, item in enumerate(items, 1):
            ck = f"{model}|{item['id']}"
            if ck in cache:
                continue
            cache[ck] = judge_once(llm, item)
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"[{jname} {n}/{len(items)}] {item['id']}", flush=True)
            time.sleep(args.delay)

    # ---- tables ----
    rows = []
    for item in items:
        cond = key.get(item["id"], "?")
        for jname, model in judges.items():
            s = cache[f"{model}|{item['id']}"]
            rows.append({"item": item["id"], "form": item["form_id"], "condition": cond,
                         "judge": jname, **{c: s[c] for c in CRITERIA}, "reason": s["reason"]})
    with open(d / "llm_judge_items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    conds = sorted({r["condition"] for r in rows})
    means = {}
    for jn in list(judges) + ["both"]:
        for c in conds:
            sel = [r for r in rows if r["condition"] == c and (jn == "both" or r["judge"] == jn)]
            means[(jn, c)] = {k: mean(r[k] for r in sel) for k in CRITERIA}
            means[(jn, c)]["overall"] = mean(means[(jn, c)][k] for k in CRITERIA)

    # paired by form: RAG better / tie / worse (average of both judges and all criteria)
    per_form = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_form[r["form"]][r["condition"]].append(mean(r[k] for k in CRITERIA))
    better = tie = worse = 0
    for form, byc in per_form.items():
        if "llm_rag" in byc and "llm_only" in byc:
            diff = mean(byc["llm_rag"]) - mean(byc["llm_only"])
            if abs(diff) < 1e-9:
                tie += 1
            elif diff > 0:
                better += 1
            else:
                worse += 1

    # agreement between the two judges over every (item, criterion) pair
    a, b = [], []
    for item in items:
        sa = cache[f"{judges['judge_A']}|{item['id']}"]
        sb = cache[f"{judges['judge_B']}|{item['id']}"]
        for c in CRITERIA:
            a.append(sa[c])
            b.append(sb[c])
    agree = {
        "n_pairs": len(a),
        "exact": mean(1 if x == y else 0 for x, y in zip(a, b)),
        "within_one": mean(1 if abs(x - y) <= 1 else 0 for x, y in zip(a, b)),
        "kappa": kappa(a, b),
        "kappa_quadratic": kappa(a, b, weighted=True),
    }

    result = {
        "note": "LLM-as-judge ratings, not human ratings",
        "judges": judges,
        "n_items": len(items),
        "means": {f"{j}|{c}": v for (j, c), v in means.items()},
        "rag_vs_alone_per_form": {"rag_better": better, "tie": tie, "rag_worse": worse},
        "judge_agreement": agree,
    }
    (d / "llm_judge_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = ["# Explanation quality: LLM alone vs LLM + RAG", "",
          "**LLM-as-judge ratings (not human ratings).** Two judge models scored each blinded item "
          "1 to 5.", "",
          f"Judges: A = `{judges['judge_A']}`, B = `{judges['judge_B']}`. Items: {len(items)}.", "",
          "## Mean scores (both judges)", "",
          "| condition | faithfulness | clarity | usefulness | overall |", "|---|---|---|---|---|"]
    for c in conds:
        m = means[("both", c)]
        md.append(f"| {c} | {m['faithfulness']:.2f} | {m['clarity']:.2f} | {m['usefulness']:.2f} | {m['overall']:.2f} |")
    md += ["", "## Per judge (overall)", "", "| judge | " + " | ".join(conds) + " |",
           "|---|" + "---|" * len(conds)]
    for jn in judges:
        md.append(f"| {jn} | " + " | ".join(f"{means[(jn, c)]['overall']:.2f}" for c in conds) + " |")
    md += ["", "## RAG vs alone, per form", "",
           f"RAG better: {better}, tie: {tie}, RAG worse: {worse}", "",
           "## Agreement between the two judges", "",
           f"- exact agreement: {agree['exact']:.2f}",
           f"- within one point: {agree['within_one']:.2f}",
           f"- Cohen's kappa: {agree['kappa']:.2f} ({landis_koch(agree['kappa'])})",
           f"- quadratic-weighted kappa: {agree['kappa_quadratic']:.2f} ({landis_koch(agree['kappa_quadratic'])})",
           f"- pairs compared: {agree['n_pairs']}", ""]
    (d / "llm_judge_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()