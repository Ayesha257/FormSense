"""Synthetic form factory. Every form comes with exact ground truth (fields, issues, capture homography).

  python scripts/get_fonts.py ; python -m formsense.generator.layout
  python -m formsense.generator.generate --count 300 --split dev  --seed 1      --out data/synthetic/dev
  python -m formsense.generator.generate --count 60  --split test --seed 100000 --out data/synthetic/test   # held out, never tune on it
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, pathlib, random
from typing import Optional
from formsense.schema import FieldResult, FieldStatus
from formsense.generator import data as D, errors as E, render as R, augment as A
from formsense.rules.reference_validator import load_template, normalise

ROOT = pathlib.Path(__file__).resolve().parents[2]
ERR_NAMES = list(E.ERRORS)


def base_state(r, today, tpl) -> E.FormState:
    p = D.make_person(r, today); st = r.choices(D.DATE_STYLES, [.7, .2, .1])[0]
    text = dict(p["text"], date_of_birth=D.fmt_date(p["dob"], st), declaration_date=D.fmt_date(p["decl"], st))
    ticks = {f["name"]: False for f in tpl["fields"] if f["type"] == "checkbox"}
    ticks["gender_" + p["gender"]] = True; ticks[r.choice(D.PROGRAMS)] = True
    ink = r.choice(R.INK)
    style = dict(font=r.choice(R.hand_fonts()), ink=ink, sig_ink=ink if r.random() < .7 else r.choice(R.INK[:3]), size_mul=r.uniform(.85, 1.05),
                 tick=r.choice(["tick", "tick", "cross", "fill"]), date_style=st)
    return E.FormState(text=text, ticks=ticks, signature="full", style=style, today=today)


def build_state(r, today, tpl, cycle_pos, err_weights=(.2, .35, .28, .17), benign_p=.5):
    s = base_state(r, today, tpl); touched: set = set()
    n = r.choices([0, 1, 2, 3], err_weights)[0]
    if n:
        order = [ERR_NAMES[cycle_pos % len(ERR_NAMES)]] + r.sample(ERR_NAMES, len(ERR_NAMES))  # first one cycles => balanced coverage
        done = 0
        for nm in order:
            if nm in s.applied and nm not in ("missing_text", "unreadable", "correction"): continue
            if E.try_apply(s, nm, r, touched, E.ERRORS): done += 1
            if done == n: break
    if r.random() < benign_p:
        for nm in r.sample(list(E.BENIGN), r.randint(1, 2)): E.try_apply(s, nm, r, touched, E.BENIGN)
    return s, n


def ground_truth(s: E.FormState, tpl, img, form_id, split, seed, image_name, cap, profile, labels):
    fields = []
    for f in tpl["fields"]:
        fid, ft = f["name"], f["kind"]; bbox = tuple(f["bbox"]); st = s.status.get(fid, "OK")
        ink = round(R.ink_ratio(img, bbox), 5)
        chk = bool(s.ticks[fid]) if ft == "checkbox" else None
        if ft in ("checkbox", "signature"): raw, val = "", None
        else:
            raw = s.text.get(fid, ""); val = None if st != "OK" else normalise(ft, raw)
        fields.append(FieldResult(name=fid, bbox=bbox, raw_text=raw, value=val, ink_ratio=ink, checked=chk, status=FieldStatus(st.lower())).model_dump(mode="json"))
    box = {f["name"]: f["bbox"] for f in tpl["fields"]}
    for i in s.issues:
        i.evidence = [tuple(box[x]) for x in i.fields]
    return dict(form_id=form_id, split=split, seed=seed, reference_date=s.today.isoformat(), template=tpl["name"], image=image_name,
                capture=dict(profile=profile, homography_template_to_image=cap), style=s.style, labels_printed=labels,
                injectors=s.applied, benign=s.benign, defects={k: v for k, v in s.defects.items()},
                fields=fields, issues=[i.model_dump(mode="json") for i in s.issues])


def build_form(i, split="dev", seed=1, today: Optional[dt.date] = None, cycle_pos=0, capture=True, variants=False, tpl=None, draw=True):
    tpl = tpl or load_template(); today = today or dt.date.today()
    r = random.Random(f"{split}-{seed}-{i}")
    s, n = build_state(r, today, tpl, cycle_pos)
    img, labels = R.blank_form(tpl, r, variants)
    if draw: R.fill_form(img, tpl, s, r)
    form_id = f"{split}_{i:04d}"
    prof = A.pick_profile(r) if capture else "none"
    out, H = A.capture(img, prof, r) if capture else (img, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    name = f"{form_id}.{'png' if prof.startswith('scan') or prof == 'none' else 'jpg'}"
    return ground_truth(s, tpl, img, form_id, split, seed, name, H, prof, labels), out, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--count", type=int, default=20); ap.add_argument("--split", default="dev")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="data/synthetic/dev")
    ap.add_argument("--today", default=None, help="reference date YYYY-MM-DD (default today)")
    ap.add_argument("--no-capture", action="store_true", help="output the clean template aligned image only")
    ap.add_argument("--label-variants", action="store_true", help="randomly print synonym labels (D.O.B, E-mail, ...)")
    a = ap.parse_args(); out = ROOT / a.out; (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today(); tpl = load_template(); rows = []; cyc = 0
    for i in range(a.count):
        gt, img, n = build_form(i, a.split, a.seed, today, cyc, not a.no_capture, a.label_variants, tpl); cyc += 1 if n else 0
        img.save(out / "images" / gt["image"], quality=92)
        (out / "labels" / f"{gt['form_id']}.json").write_text(json.dumps(gt, indent=1))
        rows.append(dict(form_id=gt["form_id"], split=a.split, image=gt["image"], capture=gt["capture"]["profile"], n_issues=len(gt["issues"]),
                         issue_types="|".join(x["type"] for x in gt["issues"]), injectors="|".join(gt["injectors"]), benign="|".join(gt["benign"])))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} forms to {out}")


if __name__ == "__main__":
    main()
