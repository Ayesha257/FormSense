"""Combine two raters' scores: agreement (Cohen's kappa) and LLM-alone vs. LLM+RAG means.

    python scripts/rating_report.py eval/rating/ratings_ayesha.json eval/rating/ratings_easha.json

Reads rating_set.json and rating_key.json from eval/rating/ (override with --dir).
Writes results.md and results.json next to them and prints the tables.
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense import rating as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratings", nargs=2, help="the two raters' ratings_<name>.json files")
    ap.add_argument("--dir", default=str(R.RATING_DIR), help="folder with rating_set.json and rating_key.json")
    args = ap.parse_args()

    folder = Path(args.dir)
    items = R.load_items(folder / "rating_set.json")
    key = R.load_key(folder / "rating_key.json")
    docs = [R.load_ratings(Path(p)) for p in args.ratings]
    names = [d.get("rater") or Path(p).stem for d, p in zip(docs, args.ratings)]
    for n, d in zip(names, docs):
        done, total = R.progress(items, d)
        print(f"{n}: {done}/{total} items rated")
        if done < total:
            print("  (only items rated by both raters count toward agreement)")
    ids = [i["id"] for i in items]
    agree = R.agreement(docs[0], docs[1], ids)
    if agree["n_items"] < 2:
        sys.exit("Fewer than 2 items rated by both raters: nothing to compare yet.")
    means = R.condition_means(items, key, docs)
    md = R.format_report(agree, means, names)
    (folder / "results.md").write_text(md, encoding="utf-8")
    (folder / "results.json").write_text(json.dumps({"raters": names, "agreement": agree, "means": means},
                                                    indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"Saved {folder / 'results.md'} and results.json")


if __name__ == "__main__":
    main()
