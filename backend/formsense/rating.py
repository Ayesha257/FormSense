"""Human rating of LLM explanations: data files, agreement (Cohen's kappa) and condition means.

Roadmap evaluation row "Explanation quality: LLM alone vs. LLM + RAG": about 30
real outputs, each rated 1-5 for faithfulness, clarity and usefulness by two
independent raters (Ayesha and Easha), plus agreement between them.

Files (all JSON, all in eval/rating/ by default):
  rating_set.json    what raters see: blinded items, no condition
  rating_key.json    item id -> condition ("llm_only" | "llm_rag"), kept away from raters
  ratings_<name>.json  one rater's scores, saved after every item
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

DIMENSIONS = ("faithfulness", "clarity", "usefulness")
SCALE = (1, 2, 3, 4, 5)
CONDITIONS = ("llm_only", "llm_rag")
CONDITION_NAMES = {"llm_only": "LLM alone", "llm_rag": "LLM + RAG"}
RATING_DIR = Path(__file__).resolve().parents[1] / "eval" / "rating"

RUBRIC = {
    "faithfulness": "Does the explanation say only what the audit facts support? "
                    "1 = invents or contradicts facts, 3 = mostly right with a slip, 5 = every claim is supported.",
    "clarity": "Could an applicant understand it on first read? "
               "1 = confusing or jargon-heavy, 3 = understandable with effort, 5 = plain and well organised.",
    "usefulness": "Does it tell the applicant exactly what to fix and where to look? "
                  "1 = no actionable advice, 3 = some advice, 5 = every problem has a clear next step.",
}


# ---------------- files ----------------

def _write_json(path: Path, data) -> None:
    """Atomic write: a crash mid-save must not destroy a rater's work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "rater"


def save_set(path: Path, items: list[dict], key: dict[str, str], key_path: Path) -> None:
    _write_json(path, {"version": 1, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "items": items})
    _write_json(key_path, key)


def load_items(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("rating set has duplicate item ids")
    for i in items:
        if not i.get("explanation"):
            raise ValueError(f"item {i['id']} has no explanation")
    return items


def load_key(path: Path) -> dict[str, str]:
    key = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = {v for v in key.values() if v not in CONDITIONS}
    if bad:
        raise ValueError(f"unknown condition(s) in key: {sorted(bad)}")
    return key


def ratings_path(name: str, folder: Path = RATING_DIR) -> Path:
    return Path(folder) / f"ratings_{slug(name)}.json"


def load_ratings(path: Path) -> dict:
    """{"rater": name, "ratings": {item id: {dimension: 1-5, ..., "comment": str}}}"""
    path = Path(path)
    if not path.exists():
        return {"rater": "", "ratings": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.setdefault("ratings", {})
    return doc


def is_complete(entry: dict) -> bool:
    return all(entry.get(d) in SCALE for d in DIMENSIONS)


def save_rating(path: Path, rater: str, item_id: str, scores: dict, comment: str = "") -> dict:
    """Validate and store one item's scores. Raises ValueError on a bad score."""
    for d in DIMENSIONS:
        if scores.get(d) not in SCALE:
            raise ValueError(f"{d} must be an integer from 1 to 5, got {scores.get(d)!r}")
    doc = load_ratings(path)
    doc["rater"] = rater
    doc["ratings"][item_id] = {**{d: int(scores[d]) for d in DIMENSIONS}, "comment": comment.strip(),
                               "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json(path, doc)
    return doc


def progress(items: Sequence[dict], doc: dict) -> tuple[int, int]:
    done = sum(1 for i in items if is_complete(doc["ratings"].get(i["id"], {})))
    return done, len(items)


# ---------------- agreement ----------------

def cohen_kappa(a: Sequence[int], b: Sequence[int], weights: Optional[str] = None,
                labels: Sequence[int] = SCALE) -> Optional[float]:
    """Cohen's kappa for two raters. weights: None (exact), "linear" or "quadratic".

    Ordinal 1-5 scores deserve quadratic weights: disagreeing by 1 is milder than by 4.
    Returns None when kappa is undefined (fewer than 2 pairs, or both raters used one single
    score everywhere so chance agreement is already 100%).
    """
    if len(a) != len(b):
        raise ValueError("raters must score the same items")
    n = len(a)
    if n < 2:
        return None
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:       # disagreement weight
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == "linear" else d * d if weights == "quadratic" else _bad(weights)

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    den = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if den == 0:
        return None
    return 1.0 - num / den


def _bad(weights):
    raise ValueError(f"weights must be None, 'linear' or 'quadratic', got {weights!r}")


def kappa_label(k: Optional[float]) -> str:
    """Landis & Koch wording, handy for the write-up."""
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return "undefined"
    for bound, name in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                        (0.8, "substantial"), (1.01, "almost perfect")):
        if k < bound:
            return name
    return "almost perfect"


def agreement(doc_a: dict, doc_b: dict, item_ids: Optional[Iterable[str]] = None) -> dict:
    """Per dimension: n, exact agreement, within one point, kappa (exact and quadratic)."""
    ra, rb = doc_a["ratings"], doc_b["ratings"]
    ids = [i for i in (item_ids if item_ids is not None else ra) if i in ra and i in rb
           and is_complete(ra[i]) and is_complete(rb[i])]
    out = {"n_items": len(ids), "dimensions": {}}
    pooled_a, pooled_b = [], []
    for d in DIMENSIONS:
        a = [ra[i][d] for i in ids]
        b = [rb[i][d] for i in ids]
        pooled_a += a
        pooled_b += b
        out["dimensions"][d] = _agree_row(a, b)
    out["overall"] = _agree_row(pooled_a, pooled_b)
    return out


def _agree_row(a: list[int], b: list[int]) -> dict:
    n = len(a)
    kq = cohen_kappa(a, b, "quadratic")
    return {"n": n,
            "exact": round(sum(x == y for x, y in zip(a, b)) / n, 3) if n else None,
            "within_one": round(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / n, 3) if n else None,
            "kappa": _r(cohen_kappa(a, b)), "kappa_quadratic": _r(kq),
            "interpretation": kappa_label(kq)}


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 3)


# ---------------- condition comparison ----------------

def condition_means(items: Sequence[dict], key: dict[str, str], docs: Sequence[dict]) -> dict:
    """Mean score per condition and dimension, averaging every complete rating from every rater.

    Also pairs the two conditions per form: how often did RAG beat, tie or lose to LLM alone?
    """
    sums: dict = {c: {d: [] for d in DIMENSIONS} for c in CONDITIONS}
    per_item: dict[str, dict[str, float]] = {}
    for it in items:
        cond = key.get(it["id"])
        if cond is None:
            continue
        got = [doc["ratings"][it["id"]] for doc in docs
               if is_complete(doc["ratings"].get(it["id"], {}))]
        if not got:
            continue
        for d in DIMENSIONS:
            sums[cond][d].extend(r[d] for r in got)
        per_item[it["id"]] = {d: sum(r[d] for r in got) / len(got) for d in DIMENSIONS}

    out = {"conditions": {}, "paired": {}}
    for c in CONDITIONS:
        out["conditions"][c] = {"n_ratings": len(sums[c][DIMENSIONS[0]]),
                                **{d: _mean(sums[c][d]) for d in DIMENSIONS}}
        vals = [v for d in DIMENSIONS for v in sums[c][d]]
        out["conditions"][c]["overall"] = _mean(vals)

    by_form: dict[str, dict[str, dict]] = defaultdict(dict)
    for it in items:
        if it["id"] in per_item and it["id"] in key:
            by_form[it["form_id"]][key[it["id"]]] = per_item[it["id"]]
    pairs = [v for v in by_form.values() if all(c in v for c in CONDITIONS)]
    for d in DIMENSIONS:
        diffs = [v["llm_rag"][d] - v["llm_only"][d] for v in pairs]
        out["paired"][d] = {"forms": len(diffs),
                            "rag_better": sum(x > 0 for x in diffs),
                            "tie": sum(x == 0 for x in diffs),
                            "rag_worse": sum(x < 0 for x in diffs),
                            "mean_difference": _mean(diffs)}
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 2) if xs else None


def format_report(agree: dict, means: dict, names: Sequence[str]) -> str:
    """Markdown tables for the final results slide."""
    L = [f"# Explanation ratings: {' and '.join(names)}", "",
         f"{agree['n_items']} items rated by both raters.", "",
         "## Mean score (1-5) by condition", "",
         "| Condition | Faithfulness | Clarity | Usefulness | Overall | Ratings |", "| --- | --- | --- | --- | --- | --- |"]
    for c in CONDITIONS:
        m = means["conditions"][c]
        L.append(f"| {CONDITION_NAMES[c]} | {m['faithfulness']} | {m['clarity']} | {m['usefulness']} | "
                 f"{m['overall']} | {m['n_ratings']} |")
    L += ["", "## RAG vs. LLM alone, paired by form", "",
          "| Dimension | Forms | RAG better | Tie | RAG worse | Mean difference |", "| --- | --- | --- | --- | --- | --- |"]
    for d in DIMENSIONS:
        p = means["paired"][d]
        L.append(f"| {d} | {p['forms']} | {p['rag_better']} | {p['tie']} | {p['rag_worse']} | {p['mean_difference']} |")
    L += ["", "## Agreement between the two raters", "",
          "| Dimension | Items | Exact | Within 1 | Kappa | Weighted kappa | Reading |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for d, r in [*agree["dimensions"].items(), ("overall", agree["overall"])]:
        L.append(f"| {d} | {r['n']} | {r['exact']} | {r['within_one']} | {r['kappa']} | "
                 f"{r['kappa_quadratic']} | {r['interpretation']} |")
    L += ["", "Weighted kappa uses quadratic weights, appropriate for ordinal 1-5 scores. "
              "Reading follows Landis and Koch (1977)."]
    return "\n".join(L) + "\n"
