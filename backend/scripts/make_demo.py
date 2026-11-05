"""Create a demo template + fake filled-form photos with ground truth.

    python scripts/make_demo.py              # 3 fixed cases + 12 random ones
    python scripts/make_demo.py --random 30  # more random ones

Writes data/templates/demo/, so the vision step, the baseline and the app can
be tried before the real admission form is designed. Each sample photo
<name>.jpg gets a <name>.truth.json with the correct answer for every field.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from formsense.schema import FieldType  # noqa: E402
from formsense.vision.demo_form import blank_form, filled_form, phone_photo, template  # noqa: E402

OUT = BACKEND / "data" / "templates" / "demo"
SAMPLES = OUT / "samples"

CASES = {
    # name: (values, ticked boxes, signed)
    "complete": ({"full_name": "Ali Raza", "father_name": "Raza Khan", "cnic": "35202-1234567-1",
                  "date_of_birth": "12/03/2005", "declaration_date": "01/11/2026"},
                 {"program_bscs"}, True),
    "no_signature_no_cnic": ({"full_name": "Sara Ahmed", "father_name": "Ahmed Ali",
                              "date_of_birth": "05/08/2006", "declaration_date": "02/11/2026"},
                             {"program_bsai"}, False),
    "two_programs": ({"full_name": "Hina Malik", "father_name": "Tariq Malik",
                      "cnic": "61101-7654321-2", "date_of_birth": "21/01/2005",
                      "declaration_date": "03/11/2026"},
                     {"program_bscs", "program_bsse"}, True),
}

FIRST = ["Ali", "Sara", "Hina", "Usman", "Ayesha", "Bilal", "Fatima", "Hamza", "Zainab",
         "Omar", "Maryam", "Saad", "Iqra", "Danish", "Noor", "Fahad", "Areeba", "Talha"]
LAST = ["Raza", "Khan", "Ahmed", "Malik", "Iqbal", "Sheikh", "Butt", "Qureshi", "Chaudhry",
        "Siddiqui", "Hussain", "Mirza", "Anwar", "Javed"]
PROGRAMS = ["program_bscs", "program_bsse", "program_bsai"]


def random_case(rng: random.Random):
    last = rng.choice(LAST)
    values = {
        "full_name": f"{rng.choice(FIRST)} {last}",
        "father_name": f"{rng.choice(FIRST)} {last}",
        "cnic": f"{rng.randint(10000, 99999)}-{rng.randint(1000000, 9999999)}-{rng.randint(1, 9)}",
        "date_of_birth": f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2003, 2008)}",
        "declaration_date": f"{rng.randint(1, 28):02d}/{rng.randint(8, 11):02d}/2026",
    }
    for k in list(values):            # each text field left empty 20% of the time
        if rng.random() < 0.2:
            del values[k]
    n_ticks = rng.choices([0, 1, 2], weights=[1, 6, 2])[0]
    ticks = set(rng.sample(PROGRAMS, n_ticks))
    signed = rng.random() < 0.7
    return values, ticks, signed


def truth(values, ticks, signed):
    rows = []
    for f in template().fields:
        if f.type == FieldType.CHECKBOX:
            rows.append({"name": f.name, "status": "ok", "checked": f.name in ticks})
        elif f.type == FieldType.SIGNATURE:
            rows.append({"name": f.name, "status": "ok" if signed else "missing"})
        else:
            v = values.get(f.name, "")
            rows.append({"name": f.name, "status": "ok" if v else "missing", "text": v})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=12, help="number of random samples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if SAMPLES.exists():
        shutil.rmtree(SAMPLES)
    SAMPLES.mkdir(parents=True)
    cv2.imwrite(str(OUT / "demo_blank.png"), blank_form())
    t = template()
    t.image_path = "data/templates/demo/demo_blank.png"
    (OUT / "demo_template.json").write_text(json.dumps(t.model_dump(mode="json"), indent=2))

    rng = random.Random(args.seed)
    cases = dict(CASES)
    for i in range(args.random):
        cases[f"random_{i + 1:02d}"] = random_case(rng)

    for i, (name, (values, ticks, signed)) in enumerate(cases.items()):
        photo = phone_photo(filled_form(values, ticks, signed), seed=i)
        cv2.imwrite(str(SAMPLES / f"{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (SAMPLES / f"{name}.truth.json").write_text(json.dumps(truth(values, ticks, signed), indent=2))
    print(f"Demo template and {len(cases)} samples (with ground truth) written to "
          f"{OUT.relative_to(BACKEND).as_posix()}")


if __name__ == "__main__":
    main()
