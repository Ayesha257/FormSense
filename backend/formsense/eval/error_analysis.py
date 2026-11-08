import csv, difflib, json
from collections import Counter
from pathlib import Path

LABELS = Path("data/synthetic/test/labels")
RES = Path("data/eval_results")


def st(x):
    return str(x).split(".")[-1].lower()


def sim(a, b):
    return difflib.SequenceMatcher(None, (a or "").strip().lower(), (b or "").strip().lower()).ratio()


def bucket(gt, pr):
    gs, ps = st(gt.get("status")), st(pr.get("status"))
    if "signature" in gt["name"]:
        return None if gs == ps else ("signature_missed" if gs == "missing" else "signature_false_alarm")
    if gt.get("checked") is not None or pr.get("checked") is not None:
        return None if bool(gt.get("checked")) == bool(pr.get("checked")) else "checkbox_wrong"
    if gs == "missing" and ps != "missing":
        return "false_present"
    if gs != "missing" and ps == "missing":
        return "false_missing"
    if gs != "missing" and ps == "unreadable":
        return "flagged_unreadable"
    if gs != "missing" and sim(gt.get("raw_text"), pr.get("raw_text")) < 0.9:
        return "low_conf_ocr" if (pr.get("ocr_confidence") or 1) < 0.6 else "ocr_text_error"
    return None


rows, counts = [], {"baseline": Counter(), "full": Counter()}
for lp in sorted(LABELS.glob("*.json")):
    doc = json.loads(lp.read_text(encoding="utf-8"))
    fid = doc.get("form_id", lp.stem)
    gts = {f["name"]: f for f in doc["fields"]}
    for system in counts:
        rp = RES / f"{system}_{fid}.json"
        if not rp.exists():
            continue
        for pr in json.loads(rp.read_text(encoding="utf-8")):
            gt = gts.get(pr["name"])
            b = bucket(gt, pr) if gt else None
            if b:
                counts[system][b] += 1
                rows.append([system, fid, pr["name"], b, gt.get("raw_text"), pr.get("raw_text"), pr.get("ocr_confidence")])

with open(RES / "error_analysis.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["system", "form", "field", "bucket", "truth", "predicted", "ocr_conf"])
    w.writerows(rows)
for s, c in counts.items():
    print(s, dict(c.most_common()))