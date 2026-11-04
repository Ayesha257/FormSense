"""Create a demo template + a few fake filled-form photos.

    python scripts/make_demo.py

Writes data/templates/demo/ so the vision step and the Streamlit app can be
tried before the real admission form is designed.
"""
import json
import sys
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

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


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "demo_blank.png"), blank_form())
    t = template()
    t.image_path = "data/templates/demo/demo_blank.png"
    (OUT / "demo_template.json").write_text(json.dumps(t.model_dump(mode="json"), indent=2))
    for i, (name, (values, ticks, signed)) in enumerate(CASES.items()):
        photo = phone_photo(filled_form(values, ticks, signed), seed=i)
        cv2.imwrite(str(SAMPLES / f"{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Demo template and {len(CASES)} samples written to {OUT.relative_to(BACKEND)}")


if __name__ == "__main__":
    main()
