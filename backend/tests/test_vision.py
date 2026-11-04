"""Vision tests on a fake form photographed at an angle (detectors only, no OCR)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense.vision.demo_form import blank_form, filled_form, phone_photo, template  # noqa: E402

from formsense.schema import FieldStatus  # noqa: E402
from formsense.vision.align import align_to_template  # noqa: E402
from formsense.vision.pipeline import analyze_page  # noqa: E402

VALUES = {"full_name": "Ali Raza", "father_name": "Raza Khan",
          "date_of_birth": "12/03/2005", "declaration_date": "01/11/2026"}
# cnic left empty on purpose


def _run(seed=0, ticks=frozenset({"program_bscs"}), signed=False):
    photo = phone_photo(filled_form(VALUES, set(ticks), signed), seed=seed)
    out = analyze_page(photo, template(), blank_form(), ocr=None)
    return {r.name: r for r in out.fields}, out


def test_alignment_recovers_template_frame():
    blank = blank_form()
    photo = phone_photo(filled_form(VALUES, {"program_bscs"}, True), seed=1)
    res = align_to_template(photo, blank)
    assert res.ok and res.inliers >= 50
    # after alignment the printed page border should sit where the template has it
    diff = np.abs(res.image.astype(int) - blank.astype(int))
    assert (diff > 80).mean() < 0.05


def test_missing_text_and_signature_detected():
    r, _ = _run(seed=2, signed=False)
    assert r["cnic"].status == FieldStatus.MISSING
    assert r["signature"].status == FieldStatus.MISSING
    for name in VALUES:
        assert r[name].status == FieldStatus.OK, name


def test_signature_present():
    r, _ = _run(seed=3, signed=True)
    assert r["signature"].status == FieldStatus.OK


def test_checkboxes():
    r, _ = _run(seed=4, ticks={"program_bscs", "program_bsai"})
    assert r["program_bscs"].checked is True
    assert r["program_bsai"].checked is True
    assert r["program_bsse"].checked is False


def test_many_photo_angles():
    for seed in range(5, 10):
        r, out = _run(seed=seed, signed=True)
        assert out.alignment.ok, seed
        assert r["cnic"].status == FieldStatus.MISSING, seed
        assert r["full_name"].status == FieldStatus.OK, seed
        assert r["program_bscs"].checked and not r["program_bsse"].checked, seed
