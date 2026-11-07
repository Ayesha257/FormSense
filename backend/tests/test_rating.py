"""Rating tool: kappa, agreement, condition means, files, and the offline set builder."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense import rating as R  # noqa: E402


def doc(rater, scores: dict) -> dict:
    """scores: {item id: (faithfulness, clarity, usefulness)}"""
    return {"rater": rater, "ratings": {i: dict(zip(R.DIMENSIONS, s), comment="") for i, s in scores.items()}}


# ---------------- kappa ----------------

def test_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert R.cohen_kappa(a, a) == pytest.approx(1.0)
    assert R.cohen_kappa(a, a, "quadratic") == pytest.approx(1.0)


def test_kappa_known_textbook_value():
    # classic 2x2 example: 20/5/10/15 -> po = 0.7, pe = 0.5, kappa = 0.4
    a = [1] * 25 + [2] * 25
    b = [1] * 20 + [2] * 5 + [1] * 10 + [2] * 15
    assert R.cohen_kappa(a, b, labels=[1, 2]) == pytest.approx(0.4)


def test_kappa_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(3)
    a = [rng.randint(1, 5) for _ in range(60)]
    b = [min(5, max(1, x + rng.choice([-1, 0, 0, 1]))) for x in a]
    for w in (None, "linear", "quadratic"):
        assert R.cohen_kappa(a, b, w) == pytest.approx(sk.cohen_kappa_score(a, b, weights=w, labels=[1, 2, 3, 4, 5]))


def test_quadratic_kappa_is_kinder_to_near_misses():
    a = [1, 2, 3, 4, 5] * 4
    near = [min(5, x + 1) for x in a]
    far = [6 - x for x in a]
    assert R.cohen_kappa(a, near, "quadratic") > R.cohen_kappa(a, far, "quadratic")


def test_kappa_undefined_cases():
    assert R.cohen_kappa([3], [3]) is None                 # too few pairs
    assert R.cohen_kappa([4, 4, 4], [4, 4, 4]) is None     # no variation: chance agreement is 100%
    with pytest.raises(ValueError):
        R.cohen_kappa([1, 2], [1])
    with pytest.raises(ValueError):
        R.cohen_kappa([1, 2], [1, 2], "cubic")


def test_kappa_labels():
    assert R.kappa_label(None) == "undefined"
    assert R.kappa_label(0.1) == "slight" and R.kappa_label(0.7) == "substantial" and R.kappa_label(0.95) == "almost perfect"
    assert R.kappa_label(-0.2) == "poor"


# ---------------- agreement + means ----------------

def test_agreement_counts_only_items_both_raters_finished():
    a = doc("a", {"i1": (5, 4, 3), "i2": (2, 2, 2), "i3": (4, 4, 4)})
    b = doc("b", {"i1": (5, 4, 4), "i2": (2, 3, 2)})
    b["ratings"]["i3"] = {"faithfulness": 4}       # incomplete: ignored
    out = R.agreement(a, b)
    assert out["n_items"] == 2
    f = out["dimensions"]["faithfulness"]
    assert f["exact"] == 1.0 and f["n"] == 2
    assert out["dimensions"]["clarity"]["within_one"] == 1.0
    assert out["overall"]["n"] == 6


ITEMS = [{"id": f"i{n}", "form_id": f"f{n // 2}"} for n in range(4)]
KEY = {"i0": "llm_only", "i1": "llm_rag", "i2": "llm_only", "i3": "llm_rag"}


def test_condition_means_and_pairing():
    a = doc("a", {"i0": (3, 3, 3), "i1": (5, 4, 5), "i2": (4, 4, 4), "i3": (4, 4, 4)})
    b = doc("b", {"i0": (3, 3, 3), "i1": (5, 5, 5), "i2": (4, 4, 4), "i3": (3, 3, 3)})
    m = R.condition_means(ITEMS, KEY, [a, b])
    assert m["conditions"]["llm_only"]["faithfulness"] == 3.5
    assert m["conditions"]["llm_rag"]["faithfulness"] == 4.25
    assert m["conditions"]["llm_only"]["n_ratings"] == 4
    p = m["paired"]["faithfulness"]
    assert (p["forms"], p["rag_better"], p["rag_worse"]) == (2, 1, 1)   # f0: 5 vs 3, f1: 3.5 vs 4


def test_format_report_has_all_tables():
    a = doc("a", {"i0": (3, 3, 3), "i1": (5, 4, 5), "i2": (4, 4, 4), "i3": (4, 4, 4)})
    b = doc("b", {"i0": (2, 3, 3), "i1": (5, 5, 4), "i2": (4, 5, 4), "i3": (3, 3, 3)})
    md = R.format_report(R.agreement(a, b), R.condition_means(ITEMS, KEY, [a, b]), ["a", "b"])
    assert "LLM alone" in md and "LLM + RAG" in md and "Weighted kappa" in md and "overall" in md


# ---------------- files ----------------

def test_save_and_reload_rating(tmp_path):
    p = R.ratings_path("Ayesha K.", tmp_path)
    assert p.name == "ratings_ayesha_k.json"
    R.save_rating(p, "Ayesha K.", "item_001", {"faithfulness": 5, "clarity": 4, "usefulness": 3}, " nice ")
    R.save_rating(p, "Ayesha K.", "item_002", {"faithfulness": 1, "clarity": 1, "usefulness": 2})
    d = R.load_ratings(p)
    assert d["rater"] == "Ayesha K." and d["ratings"]["item_001"]["comment"] == "nice"
    assert R.progress([{"id": "item_001"}, {"id": "item_002"}, {"id": "item_003"}], d) == (2, 3)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("scores", [{"faithfulness": 6, "clarity": 3, "usefulness": 3},
                                    {"faithfulness": 3, "clarity": 3},
                                    {"faithfulness": 3, "clarity": 0, "usefulness": 3}])
def test_save_rating_rejects_bad_scores(tmp_path, scores):
    with pytest.raises(ValueError):
        R.save_rating(tmp_path / "r.json", "x", "item_001", scores)
    assert not (tmp_path / "r.json").exists()


def test_load_ratings_missing_file_is_empty(tmp_path):
    assert R.load_ratings(tmp_path / "none.json")["ratings"] == {}


def test_load_items_validation(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"items": [{"id": "a", "explanation": "x"}, {"id": "a", "explanation": "y"}]}))
    with pytest.raises(ValueError, match="duplicate"):
        R.load_items(p)
    p.write_text(json.dumps({"items": [{"id": "a", "explanation": ""}]}))
    with pytest.raises(ValueError, match="no explanation"):
        R.load_items(p)


def test_load_key_rejects_unknown_condition(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"item_001": "llm_magic"}))
    with pytest.raises(ValueError):
        R.load_key(p)


# ---------------- end to end: offline set -> two raters -> report ----------------

def run(*args):
    return subprocess.run([sys.executable, *args], cwd=BACKEND, capture_output=True, text=True, timeout=300)


def test_offline_pipeline_end_to_end(tmp_path):
    pytest.importorskip("langchain_community")
    pytest.importorskip("faiss")
    forms = tmp_path / "forms"
    g = run("-m", "formsense.generator.generate", "--count", "10", "--split", "dev", "--seed", "5", "--out", str(forms))
    assert g.returncode == 0, g.stderr
    out = tmp_path / "rating"
    m = run("scripts/make_rating_set.py", "--dir", str(forms), "--limit", "3", "--offline", "--out", str(out))
    assert m.returncode == 0, m.stderr + m.stdout
    items = R.load_items(out / "rating_set.json")
    key = R.load_key(out / "rating_key.json")
    assert len(items) == 6 and set(key) == {i["id"] for i in items}
    assert sorted(key.values()) == ["llm_only"] * 3 + ["llm_rag"] * 3
    assert "llm_" not in (out / "rating_set.json").read_text()          # raters cannot see the condition
    assert all(i["issues"] for i in items)
    by_form = {}
    for i in items:
        by_form.setdefault(i["form_id"], set()).add(key[i["id"]])
    assert all(v == {"llm_only", "llm_rag"} for v in by_form.values())

    rng = random.Random(1)
    for who in ("ana", "bea"):
        for it in items:
            R.save_rating(R.ratings_path(who, out), who, it["id"],
                          {d: rng.randint(2, 5) for d in R.DIMENSIONS})
    rep = run("scripts/rating_report.py", str(R.ratings_path("ana", out)), str(R.ratings_path("bea", out)),
              "--dir", str(out))
    assert rep.returncode == 0, rep.stderr + rep.stdout
    res = json.loads((out / "results.json").read_text())
    assert res["agreement"]["n_items"] == 6 and (out / "results.md").exists()


def test_report_refuses_when_nothing_is_rated_twice(tmp_path):
    out = tmp_path
    R.save_set(out / "rating_set.json", [{"id": "item_001", "form_id": "f", "explanation": "x", "issues": []}],
               {"item_001": "llm_only"}, out / "rating_key.json")
    R.save_rating(R.ratings_path("a", out), "a", "item_001", {d: 3 for d in R.DIMENSIONS})
    R.save_rating(R.ratings_path("b", out), "b", "item_001", {d: 3 for d in R.DIMENSIONS})
    rep = run("scripts/rating_report.py", str(R.ratings_path("a", out)), str(R.ratings_path("b", out)),
              "--dir", str(out))
    assert rep.returncode != 0 and "Fewer than 2" in (rep.stderr + rep.stdout)
