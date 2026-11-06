"""End to end: raw field readings -> extraction -> rules -> annotated fields."""
import random

import pytest

from form_factory import TODAY, clean_form, corrupted_form
from formsense.nlp.extract import ExtractionConfig
from formsense.rules.audit import audit
from formsense.rules.engine import issue_key
from formsense.schema import FieldStatus as S


def ocr_like(fields, conf):
    """Strip the ground truth down to what the vision side really delivers."""
    out = []
    for f in fields:
        upd = {"value": None, "status": S.OK}
        if f.raw_text:
            upd["ocr_confidence"] = conf
        out.append(f.model_copy(update=upd))
    return out


def test_clean_forms_pass_end_to_end(mini_tpl, mini_engine):
    rng = random.Random(0)
    for _ in range(150):
        raw = ocr_like(list(clean_form(mini_tpl, rng).values()), conf=0.96)
        res = audit(raw, mini_tpl, TODAY, engine=mini_engine)
        assert res.issues == [], [i.id for i in res.issues]
        assert all(f.status == S.OK for f in res.fields)


def test_extraction_does_not_change_which_errors_are_found(mini_tpl, mini_engine):
    """Same issues whether statuses come from the generator or are re-derived from ink and confidence."""
    rng = random.Random(1)
    for _ in range(600):
        truth = corrupted_form(mini_tpl, rng, rng.randint(1, 3))
        direct = {issue_key(i) for i in mini_engine.validate(truth, TODAY)}
        via = {issue_key(i) for i in audit(truth, mini_tpl, TODAY, engine=mini_engine).issues}
        assert via == direct, via ^ direct


def test_blank_signature_is_found_from_ink_alone(mini_tpl, mini_engine):
    raw = ocr_like(list(clean_form(mini_tpl, random.Random(2)).values()), 0.96)
    raw = [f.model_copy(update={"ink_ratio": 0.0}) if f.name == "signature" else f for f in raw]
    res = audit(raw, mini_tpl, TODAY, engine=mini_engine)
    assert [(i.id, tuple(i.fields)) for i in res.issues] == [("missing_required", ("signature",))]


def test_unsure_ocr_is_repaired_but_confident_ocr_is_not(mini_tpl, mini_engine):
    raw = ocr_like(list(clean_form(mini_tpl, random.Random(3)).values()), 0.96)
    broken = [f.model_copy(update={"raw_text": f.raw_text.replace("0", "O", 1) if "0" in f.raw_text else f.raw_text})
              if f.name == "cnic" else f for f in raw]
    cnic = next(f for f in broken if f.name == "cnic")
    if cnic.raw_text == next(f for f in raw if f.name == "cnic").raw_text:
        broken = [f.model_copy(update={"raw_text": "3520O-1234567-1"}) if f.name == "cnic" else f for f in broken]
    confident = audit(broken, mini_tpl, TODAY, engine=mini_engine)
    assert "invalid_cnic" in [i.id for i in confident.issues]            # read with 96% confidence: applicant error
    unsure = [f.model_copy(update={"ocr_confidence": 0.85}) if f.name == "cnic" else f for f in broken]
    res = audit(unsure, mini_tpl, TODAY, engine=mini_engine)
    assert "invalid_cnic" not in [i.id for i in res.issues]              # 85%: treated as an OCR slip and repaired
    assert any("repaired" in n for n in res.notes["cnic"])


def test_low_confidence_field_is_reported_unreadable_not_invalid(mini_tpl, mini_engine):
    raw = ocr_like(list(clean_form(mini_tpl, random.Random(4)).values()), 0.96)
    raw = [f.model_copy(update={"raw_text": "A8dr3ss", "ocr_confidence": 0.4}) if f.name == "address" else f for f in raw]
    res = audit(raw, mini_tpl, TODAY, engine=mini_engine)
    assert [i.id for i in res.issues] == ["unreadable_field"]


def test_output_fields_are_annotated(mini_tpl, mini_engine):
    raw = ocr_like(list(clean_form(mini_tpl, random.Random(5)).values()), 0.96)
    raw = [f.model_copy(update={"raw_text": "31/02/2005"}) if f.name == "date_of_birth" else f for f in raw]
    res = audit(raw, mini_tpl, TODAY, engine=mini_engine)
    dob = next(f for f in res.fields if f.name == "date_of_birth")
    assert dob.status == S.INVALID and dob.value is None


def test_audit_accepts_ground_truth_without_extraction(mini_tpl, mini_engine):
    truth = list(clean_form(mini_tpl, random.Random(6)).values())
    assert audit(truth, mini_tpl, TODAY, engine=mini_engine, extract=False).issues == []
