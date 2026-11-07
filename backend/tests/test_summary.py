"""UI helpers: checklist, plain summary, crops, optional-module discovery."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense import summary as S  # noqa: E402
from formsense.rag.retriever import field_labels  # noqa: E402
from formsense.schema import AuditReport, FieldResult, FieldStatus, Issue, IssueType, Severity  # noqa: E402
from formsense.vision import load_template  # noqa: E402

TEMPLATE, _ = load_template()
LABELS = field_labels()


def make_fields(**over) -> list[FieldResult]:
    out = []
    for f in TEMPLATE.fields:
        if f.type.value == "checkbox":
            r = FieldResult(name=f.name, bbox=f.bbox, checked=f.name in ("gender_male", "program_bscs"))
        elif f.type.value == "signature":
            r = FieldResult(name=f.name, bbox=f.bbox, value="signed")
        else:
            r = FieldResult(name=f.name, bbox=f.bbox, raw_text=f"text {f.name}", ocr_confidence=0.95)
        out.append(r)
    for r in out:
        if r.name in over:
            for k, v in over[r.name].items():
                setattr(r, k, v)
    return out


def issue(sev, fields, msg="problem", t=IssueType.FORMAT):
    return Issue(id="x", type=t, severity=sev, fields=fields, message=msg)


def row(rows, item):
    return next(r for r in rows if r["item"] == item)


def test_clean_form_checklist_is_all_ok():
    rows = S.checklist(make_fields(), TEMPLATE, LABELS)
    assert all(r["status"] == "ok" for r in rows)
    assert row(rows, "Gender")["detail"] == "Male"
    assert row(rows, "Program")["detail"] == "BS Computer Science"
    assert len(rows) == len(TEMPLATE.fields) - 1 - 2     # gender (2 boxes) and program (3 boxes) collapse to one row each


def test_checkbox_groups_need_exactly_one_tick():
    none = S.checklist(make_fields(program_bscs={"checked": False}), TEMPLATE, LABELS)
    assert row(none, "Program")["status"] == "missing" and row(none, "Program")["detail"] == "nothing ticked"
    two = S.checklist(make_fields(program_bsai={"checked": True}), TEMPLATE, LABELS)
    assert row(two, "Program")["status"] == "invalid" and "more than one" in row(two, "Program")["detail"]


def test_checklist_details_for_missing_unreadable_signature():
    rows = S.checklist(make_fields(
        full_name={"status": FieldStatus.MISSING, "raw_text": ""},
        address={"status": FieldStatus.UNREADABLE, "raw_text": "h0use"},
        signature={"status": FieldStatus.MISSING, "value": None}), TEMPLATE, LABELS)
    assert row(rows, "Full Name")["detail"] == "left empty"
    assert "h0use" in row(rows, "Postal Address")["detail"]
    assert row(rows, "Signature of Applicant")["detail"] == "signature box is empty"


def test_field_counts_use_items_not_boxes():
    c = S.field_counts(make_fields(full_name={"status": FieldStatus.MISSING}), TEMPLATE)
    assert c["missing"] == 1 and c["invalid"] == 0 and sum(c.values()) == len(S.checklist(make_fields(), TEMPLATE, {}))


def test_checklist_csv_roundtrip():
    text = S.checklist_csv(S.checklist(make_fields(), TEMPLATE, LABELS))
    assert text.splitlines()[0] == "section,item,status,detail" and "Full Name" in text


def test_sort_and_count_issues():
    iss = [issue(Severity.INFO, ["a"]), issue(Severity.ERROR, ["b"]), issue(Severity.WARNING, ["c"])]
    assert [i.fields[0] for i in S.sort_issues(iss)] == ["b", "c", "a"]
    assert S.issue_counts(iss) == {"error": 1, "warning": 1, "info": 1}


def test_plain_summary():
    assert "No problems found" in S.plain_summary(make_fields(), [], TEMPLATE, LABELS)
    text = S.plain_summary(make_fields(), [issue(Severity.WARNING, ["cnic"], "CNIC is short"),
                                           issue(Severity.ERROR, ["signature"], "Signature missing")], TEMPLATE, LABELS)
    assert text.startswith("This form has 1 error and 1 warning")
    assert text.index("Signature") < text.index("CNIC is short")        # errors first
    assert "**CNIC No.**" in text


def test_crop_field_clips_to_image():
    img = np.zeros((100, 200), np.uint8)
    assert S.crop_field(img, (150, 80, 100, 100), pad=10).shape == (30, 60)
    assert S.crop_field(img, (500, 500, 10, 10)).shape == (40, 120)       # fully outside: blank placeholder


def test_highlight_changes_only_the_box_area_and_skips_empty_boxes():
    img = np.full((120, 120, 3), 255, np.uint8)
    out = S.highlight(img, [(40, 40, 20, 20), (0, 0, 0, 0)])
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(out[:10, :10], img[:10, :10])
    assert np.array_equal(S.highlight(img, [(0, 0, 0, 0)]), img)


def test_png_bytes_is_a_png():
    assert S.png_bytes(np.zeros((5, 5, 3), np.uint8)).startswith(b"\x89PNG")


def test_optional_modules_absent_by_default_and_found_when_present(monkeypatch, tmp_path):
    assert S._find([("does_not_exist_anywhere", "x")]) is None
    (tmp_path / "fake_llm_mod.py").write_text("def explain(report):\n    return 'hello'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    fn = S._find([("does_not_exist_anywhere", "x"), ("fake_llm_mod", "explain")])
    assert fn(None) == "hello"


def test_explanation_text_accepts_str_or_report():
    assert S.explanation_text(None) is None and S.explanation_text("hi") == "hi"
    assert S.explanation_text(AuditReport(form_id="f", fields=[], issues=[], explanation="from report")) == "from report"
