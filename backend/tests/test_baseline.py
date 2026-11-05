"""Baseline logic tests on hand-made OCR output (no OCR model needed, runs in < 1 s)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense.baseline.ocr_regex import BaselineAuditor, group_rows  # noqa: E402
from formsense.schema import FieldSpec, FieldStatus, FieldType, FormTemplate  # noqa: E402
from formsense.vision.demo_form import template  # noqa: E402
from formsense.vision.ocr import OCRBox  # noqa: E402


def box(text, x, y, w=200, h=30, score=0.99):
    return OCRBox(text, score, x, y, x + w, y + h)


def run(boxes, tpl=None):
    auditor = BaselineAuditor(tpl or template(), reader=object())  # reader unused by extract()
    return {r.name: r for r in auditor.extract(group_rows(boxes))}


def test_rows_follow_reading_order():
    rows = group_rows([box("Ali Raza", 400, 102), box("Full Name", 80, 100),
                       box("Father's Name", 80, 200)])
    assert [r.text for r in rows] == ["Full Name Ali Raza", "Father's Name"]


def test_value_after_label_and_longest_label_first():
    r = run([box("Full Name", 80, 100), box("Ali Raza", 400, 100),
             box("Date of Birth", 80, 300), box("12/03/2005", 400, 300),
             box("Date", 80, 900), box("01/11/2026", 400, 900)])
    assert r["full_name"].raw_text == "Ali Raza" and r["full_name"].status == FieldStatus.OK
    assert r["date_of_birth"].raw_text == "12/03/2005"
    assert r["declaration_date"].raw_text == "01/11/2026"


def test_empty_field_is_missing_and_label_case_is_ignored():
    r = run([box("FULL NAME:", 80, 100), box("CNIC No.", 80, 200)])
    assert r["full_name"].status == FieldStatus.MISSING
    assert r["cnic"].status == FieldStatus.MISSING


def test_regex_fallback_finds_cnic_without_label():
    r = run([box("Full Name", 80, 100), box("Ali Raza", 400, 100),
             box("35202-1234567-1", 400, 500)])
    assert r["cnic"].raw_text == "35202-1234567-1"


def test_value_on_next_line():
    r = run([box("Father's Name", 80, 100), box("Raza Khan", 80, 150)])
    assert r["father_name"].raw_text == "Raza Khan"


def test_value_stops_at_next_label_on_same_line():
    tpl = FormTemplate(name="t", image_path="", width=1000, height=1000, fields=[
        FieldSpec(name="date_of_birth", label="Date of Birth", type=FieldType.DATE, bbox=(0, 0, 1, 1)),
        FieldSpec(name="gender", label="Gender", type=FieldType.TEXT, bbox=(0, 0, 1, 1)),
    ])
    r = run([box("Date of Birth", 50, 100), box("12/03/2005", 300, 100),
             box("Gender", 550, 100), box("Female", 750, 100)], tpl)
    assert r["date_of_birth"].raw_text == "12/03/2005"
    assert r["gender"].raw_text == "Female"


def test_low_confidence_is_unreadable():
    r = run([box("Full Name", 80, 100), box("A1i R@za", 400, 100, score=0.41)])
    assert r["full_name"].status == FieldStatus.UNREADABLE


def test_checkbox_only_from_tick_characters():
    r = run([box("v", 60, 700, w=20), box("BS Computer Science", 120, 700),
             box("BS Software Engineering", 120, 780)])
    assert r["program_bscs"].checked is True
    assert r["program_bsse"].checked is False
    assert r["program_bsai"].checked is False   # label not even read


def test_signature_is_always_assumed_present():
    r = run([box("Signature", 80, 1400)])
    assert r["signature"].status == FieldStatus.OK


@pytest.mark.slow
def test_baseline_on_demo_photo():
    """Full run with PaddleOCR on a fake photo (skipped if PaddleOCR isn't installed)."""
    pytest.importorskip("paddleocr")
    from formsense.vision.demo_form import filled_form, phone_photo
    values = {"full_name": "Ali Raza", "date_of_birth": "12/03/2005"}
    page = phone_photo(filled_form(values, {"program_bscs"}, signed=False), seed=1)
    out = BaselineAuditor(template()).run(page)
    r = {f.name: f for f in out.fields}
    assert r["full_name"].raw_text == "Ali Raza"
    assert r["cnic"].status == FieldStatus.MISSING
    assert r["signature"].status == FieldStatus.OK        # the baseline's blind spot
