"""Pairwise LLM-as-judge: for each form, which explanation is better, LLM alone or LLM + RAG?

Absolute 1-5 scoring saturates (everything gets 5), so this compares the two
explanations of the same form directly. Each comparison runs twice with the order
swapped; if the judge's pick flips with the order it counts as a tie (position bias).
These are LLM-judge results, not human ratings.

    python scripts/llm_judge_pairwise.py
    python scripts/llm_judge_pairwise.py --dir eval/rating --delay 2

Judge models: GROQ_JUDGE_A / GROQ_JUDGE_B in .env (same as llm_judge.py).
Writes into --dir: llm_judge_pairwise_cache.json (resumable),
llm_judge_pairwise_results.json, llm_judge_pairwise_results.md
Use synthetic forms only: facts are sent to a third-party API.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

CRITERIA = ("faithfulness", "clarity", "usefulness")

SYSTEM = """You compare two explanations, A and B, of the same automatic check of a university admission form.
You get the facts read from the form, the issues found, and both explanations.
For each criterion choose "A", "B" or "tie":
- faithfulness: claims are supported by the facts and issues; nothing invented; no issue omitted.
- clarity: plain English, well organised, easy for a student to follow.
- usefulness: says exactly what to fix and how (for example the exact required format), so the person can act without guessing.
Be decisive: answer "tie" only if the two are truly equal on that criterion. Do not prefer the longer one just for length.
Reply with ONLY one JSON object:
{"faithfulness": "A|B|tie", "clarity": "A|B|tie", "usefulness": "A|B|tie", "reason": "<one short sentence>"}"""


def build_prompt(facts, issues, text_a, text_b) -> str:
    iss = "\n".join(f"- [{i['severity']}] {i['message']}" for i in issues) or "(none)"
    fct = "\n".join(f"- {f}" for f in facts)
    return (f"FORM FACTS:\n{fct}\n\nISSUES FOUND:\n{iss}\n\n"
            f"EXPLANATION A:\n{text_a}\n\nEXPLANATION B:\n{text_b}")


def make_llm(model: str):
    from langchain_groq import ChatGroq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (put it in backend/.env)")
    return ChatGroq(model=model, api_key=key, temperature=0, max_tokens=3000)


def ask(llm, prompt: str, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            text = llm.invoke([("system", SYSTEM), ("human", prompt)]).content
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            data = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
            out = {}
            for c in CRITERIA:
                v = str(data[c]).strip().lower()
                out[c] = {"a": "A", "b": "B"}.get(v, "tie")
            out["reason"] = str(data.get("reason", ""))[:300]
            return out
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"pairwise judge failed: {last!r}")


def load_key(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("items", raw) if isinstance(raw, dict) else raw
    if isinstance(raw, list):
        raw = {r["id"]: r for r in raw}
    return {k: (v.get("condition", v.get("cond")) if isinstance(v, dict) else v) for k, v in raw.items()}


def cohen_kappa(a, b):
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="eval/rating")
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    d = Path(args.dir)
    items = json.loads((d / "rating_set.json").read_text(encoding="utf-8"))["items"]
    key = load_key(d / "rating_key.json")
    forms = defaultdict(dict)
    for it in items:
        forms[it["form_id"]][key[it["id"]]] = it
    forms = {f: c for f, c in forms.items() if "llm_rag" in c and "llm_only" in c}

    judges = {"judge_A": os.environ.get("GROQ_JUDGE_A", "openai/gpt-oss-120b"),
              "judge_B": os.environ.get("GROQ_JUDGE_B", "openai/gpt-oss-20b")}
    cache_path = d / "llm_judge_pairwise_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    # verdicts[judge][form][criterion] -> "rag" | "alone" | "tie"; also count order flips
    verdicts = {j: defaultdict(dict) for j in judges}
    flips = Counter()
    total = Counter()
    for jname, model in judges.items():
        llm = make_llm(model)
        for n, (form, c) in enumerate(sorted(forms.items()), 1):
            rag, alone = c["llm_rag"], c["llm_only"]
            res = {}
            for order in ("rag_first", "alone_first"):
                ck = f"{model}|{form}|{order}"
                if ck not in cache:
                    ta, tb = (rag, alone) if order == "rag_first" else (alone, rag)
                    cache[ck] = ask(llm, build_prompt(rag["facts"], rag["issues"],
                                                      ta["explanation"], tb["explanation"]))
                    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                    time.sleep(args.delay)
                res[order] = cache[ck]
            print(f"[{jname} {n}/{len(forms)}] {form}", flush=True)
            for crit in CRITERIA:
                def to_side(order):
                    v = res[order][crit]
                    if v == "tie":
                        return "tie"
                    first_is_rag = order == "rag_first"
                    return "rag" if (v == "A") == first_is_rag else "alone"
                s1, s2 = to_side("rag_first"), to_side("alone_first")
                total[jname] += 1
                if s1 == s2:
                    verdicts[jname][form][crit] = s1
                else:
                    flips[jname] += 1
                    verdicts[jname][form][crit] = "tie"   # order-dependent pick = no real preference

    # ---- aggregate ----
    out = {"note": "LLM-as-judge, pairwise with order swap; not human ratings", "judges": judges,
           "n_forms": len(forms), "per_judge": {}, "combined": {}, "order_flip_rate": {}}
    for j in judges:
        out["order_flip_rate"][j] = flips[j] / total[j] if total[j] else float("nan")
        out["per_judge"][j] = {}
        for crit in CRITERIA:
            cnt = Counter(verdicts[j][f][crit] for f in forms)
            out["per_judge"][j][crit] = {k: cnt.get(k, 0) for k in ("rag", "tie", "alone")}
    for crit in CRITERIA:
        cnt = Counter(verdicts[j][f][crit] for j in judges for f in forms)
        out["combined"][crit] = {k: cnt.get(k, 0) for k in ("rag", "tie", "alone")}
    la, lb = [], []
    for f in forms:
        for crit in CRITERIA:
            la.append(verdicts["judge_A"][f][crit])
            lb.append(verdicts["judge_B"][f][crit])
    out["judge_agreement"] = {
        "pairs": len(la),
        "exact": sum(1 for x, y in zip(la, lb) if x == y) / len(la),
        "kappa": cohen_kappa(la, lb),
    }
    (d / "llm_judge_pairwise_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# LLM alone vs LLM + RAG: pairwise comparison", "",
          "**LLM-as-judge (not human ratings).** For each form the two explanations were compared "
          "directly, in both orders. A pick that flips with the order counts as a tie.", "",
          f"Judges: A = `{judges['judge_A']}`, B = `{judges['judge_B']}`. Forms: {len(forms)}.", "",
          "## Results (both judges combined; each form is counted once per judge)", "",
          "| criterion | RAG wins | tie | LLM alone wins |", "|---|---|---|---|"]
    for crit in CRITERIA:
        c = out["combined"][crit]
        md.append(f"| {crit} | {c['rag']} | {c['tie']} | {c['alone']} |")
    md += ["", "## Per judge", "", "| judge | criterion | RAG wins | tie | alone wins |", "|---|---|---|---|---|"]
    for j in judges:
        for crit in CRITERIA:
            c = out["per_judge"][j][crit]
            md.append(f"| {j} | {crit} | {c['rag']} | {c['tie']} | {c['alone']} |")
    md += ["", "## Reliability", "",
           f"- order-flip rate (position bias): A {out['order_flip_rate']['judge_A']:.2f}, "
           f"B {out['order_flip_rate']['judge_B']:.2f}",
           f"- judge agreement: exact {out['judge_agreement']['exact']:.2f}, "
           f"Cohen's kappa {out['judge_agreement']['kappa']:.2f} "
           f"({out['judge_agreement']['pairs']} comparisons)", ""]
    (d / "llm_judge_pairwise_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()