import pytest

from formsense.nlp.extract import ExtractionConfig, enrich_fields, extract_field, extract_fields
from formsense.schema import FieldResult, FieldStatus as S

CFG = ExtractionConfig()


def fr(name, raw="", ink=0.05, conf=None, checked=None, bbox=(0, 0, 10, 10)):
    return FieldResult(name=name, bbox=bbox, raw_text=raw, ink_ratio=ink, ocr_confidence=conf, checked=checked)


def run(tpl, name, **kw):
    return extract_field(fr(name, **kw), tpl.fields[name], CFG)


# ---------- the brief's missing versus unreadable rule ----------
def test_no_ink_no_text_is_missing(tpl):
    assert run(tpl, "full_name", raw="", ink=0.0).result.status == S.MISSING


def test_ink_but_nothing_read_is_unreadable(tpl):
    e = run(tpl, "full_name", raw="", ink=0.04)
    assert e.result.status == S.UNREADABLE and e.notes


def test_ink_and_low_confidence_is_unreadable(tpl):
    e = run(tpl, "full_name", raw="Al1 Rxz", ink=0.04, conf=0.55)
    assert e.result.status == S.UNREADABLE and e.result.value is None


def test_confident_read_is_ok(tpl):
    e = run(tpl, "full_name", raw="Ali Raza", conf=0.97)
    assert e.result.status == S.OK and e.result.value == "Ali Raza"


def test_text_wins_when_ink_measurement_is_zero(tpl):
    """A default ink_ratio of 0.0 must not erase a successful OCR read."""
    e = run(tpl, "full_name", raw="Ali Raza", ink=0.0, conf=0.95)
    assert e.result.status == S.OK and any("ink" in n for n in e.notes)


# ---------- normalisation and INVALID ----------
def test_dates_become_iso(tpl):
    e = run(tpl, "date_of_birth", raw="14 Mar 2005", conf=0.95)
    assert e.result.status == S.OK and e.result.value == "2005-03-14"


def test_impossible_date_is_invalid_with_no_value(tpl):
    e = run(tpl, "date_of_birth", raw="31/02/2005", conf=0.95)
    assert e.result.status == S.INVALID and e.result.value is None


def test_bad_cnic_keeps_raw_as_value_and_is_invalid(tpl):
    e = run(tpl, "cnic", raw="3520212345671", conf=0.97)
    assert e.result.status == S.INVALID and e.result.value == "3520212345671"


def test_placeholder_is_invalid(tpl):
    assert run(tpl, "address", raw="N/A", conf=0.9).result.status == S.INVALID
    assert run(tpl, "date_of_birth", raw="-", conf=0.9).result.status == S.INVALID


def test_raw_text_is_never_modified(tpl):
    e = run(tpl, "cnic", raw="  35202-12345O7-1 ", conf=0.7 + 0.15)   # 0.85: readable, repair allowed
    assert e.result.raw_text == "  35202-12345O7-1 "


# ---------- repair is gated by OCR confidence ----------
def test_confident_letter_o_is_an_applicant_error_not_an_ocr_slip(tpl):
    e = run(tpl, "cnic", raw="35202-12345O7-1", conf=0.98)
    assert e.result.status == S.INVALID


def test_unsure_ocr_gets_repaired(tpl):
    e = run(tpl, "cnic", raw="35202-12345O7-1", conf=0.85)
    assert e.result.status == S.OK and e.result.value == "35202-1234507-1"
    assert any("repaired" in n for n in e.notes)


def test_no_confidence_means_no_repair(tpl):
    assert run(tpl, "cnic", raw="35202-12345O7-1", conf=None).result.status == S.INVALID


# ---------- checkbox and signature ----------
def test_checkbox_uses_detector_value_first(tpl):
    assert run(tpl, "gender_male", ink=0.0, checked=True).result.checked is True


def test_checkbox_falls_back_to_ink(tpl):
    assert run(tpl, "gender_male", ink=0.3).result.checked is True
    assert run(tpl, "gender_male", ink=0.01).result.checked is False


def test_signature_needs_real_ink(tpl):
    assert run(tpl, "signature", ink=0.06).result.status == S.OK
    e = run(tpl, "signature", ink=0.004)
    assert e.result.status == S.MISSING and e.notes          # a stray dot is not a signature
    assert run(tpl, "signature", ink=0.0).result.status == S.MISSING


# ---------- batch behaviour ----------
def test_unknown_field_passes_through(tpl):
    out = extract_fields([fr("not_in_template", raw="x")], tpl)
    assert out[0].result.name == "not_in_template" and out[0].notes


def test_enrich_returns_field_results(tpl):
    out = enrich_fields([fr("full_name", raw="Ali Raza", conf=0.95)], tpl)
    assert isinstance(out[0], FieldResult) and out[0].status == S.OK


def test_extractor_never_raises_on_odd_input(tpl):
    weird = fr("matric_total", raw="\u0000\u202e\U0001f600" * 5, conf=0.9)
    assert extract_fields([weird], tpl)[0].result.status in (S.INVALID, S.UNREADABLE, S.OK)
