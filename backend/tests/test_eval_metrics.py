"""Tests for formsense/eval/metrics.py."""
from __future__ import annotations

import math

import pytest

from formsense.eval.metrics import (
    PRF, char_error_rate, score_field_extraction, score_issue_detection,
    score_missing_field_detection, score_missing_vs_unreadable,
)
from formsense.schema import FieldResult, FieldStatus as S, Issue, IssueType, Severity


def fr(name, raw="", value=None, status=S.OK):
    return FieldResult(name=name, bbox=(0, 0, 1, 1), raw_text=raw, value=value if value is not None else (raw or None), status=status)


# ---------- CER ----------
def test_cer_identical_strings_is_zero():
    assert char_error_rate("Ali Raza", "Ali Raza") == 0.0


def test_cer_one_substitution():
    assert char_error_rate("Ala Raza", "Ali Raza") == pytest.approx(1 / 8)


def test_cer_empty_truth_nonempty_pred_is_capped_at_one():
    assert char_error_rate("xxx", "") == 1.0


def test_cer_both_empty_is_zero():
    assert char_error_rate("", "") == 0.0


def test_cer_is_capped_at_one_for_very_wrong_prediction():
    assert char_error_rate("completely different and much longer", "a") <= 1.0


# ---------- field extraction ----------
def test_field_extraction_exact_and_cer():
    pred = [fr("full_name", "Ali Raza"), fr("cnic", "35202-1234567-1")]
    truth = [fr("full_name", "Ali Raza"), fr("cnic", "35202-1234567-X")]
    score = score_field_extraction(pred, truth)
    assert score.n == 2 and score.exact_matches == 1
    assert score.mean_cer == pytest.approx(char_error_rate("35202-1234567-1", "35202-1234567-X") / 2)


def test_field_extraction_skips_missing_ground_truth_by_default():
    pred = [fr("address", "whatever")]
    truth = [fr("address", "", status=S.MISSING)]
    score = score_field_extraction(pred, truth)
    assert score.n == 0


def test_field_extraction_missing_prediction_counts_as_wrong():
    pred = []
    truth = [fr("full_name", "Ali Raza")]
    score = score_field_extraction(pred, truth)
    assert score.n == 1 and score.exact_matches == 0 and score.mean_cer == 1.0


def test_field_extraction_per_field_breakdown():
    pred = [fr("full_name", "Ali Raza"), fr("cnic", "bad")]
    truth = [fr("full_name", "Ali Raza"), fr("cnic", "35202-1234567-1")]
    score = score_field_extraction(pred, truth)
    assert score.per_field["full_name"].accuracy == 1.0
    assert score.per_field["cnic"].accuracy == 0.0


def test_field_extraction_can_restrict_to_a_field_subset():
    pred = [fr("full_name", "Ali Raza"), fr("cnic", "bad")]
    truth = [fr("full_name", "Ali Raza"), fr("cnic", "35202-1234567-1")]
    score = score_field_extraction(pred, truth, fields=["full_name"])
    assert score.n == 1


def test_field_extraction_nan_when_nothing_scored():
    assert math.isnan(score_field_extraction([], []).accuracy)


# ---------- missing field detection ----------
def test_missing_detection_perfect():
    truth = [fr("a", "", status=S.MISSING), fr("b", "x")]
    pred = [fr("a", "", status=S.MISSING), fr("b", "x")]
    prf = score_missing_field_detection(pred, truth)
    assert (prf.tp, prf.fp, prf.fn) == (1, 0, 0)
    assert prf.precision == 1.0 and prf.recall == 1.0 and prf.f1 == 1.0


def test_missing_detection_false_positive_and_false_negative():
    truth = [fr("a", "", status=S.MISSING), fr("b", "x")]
    pred = [fr("a", "x"), fr("b", "", status=S.MISSING)]   # swapped
    prf = score_missing_field_detection(pred, truth)
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_prf_handles_zero_division_as_nan():
    prf = PRF()
    assert math.isnan(prf.precision) and math.isnan(prf.recall) and math.isnan(prf.f1)


def test_prf_addition():
    a, b = PRF(tp=1, fp=1, fn=0), PRF(tp=2, fp=0, fn=1)
    c = a + b
    assert (c.tp, c.fp, c.fn) == (3, 1, 1)


# ---------- issue detection ----------
def issue(id_, fields, type_="inconsistency"):
    return Issue(id=id_, type=IssueType(type_), severity=Severity.ERROR, fields=fields, message="m")


def test_issue_detection_exact_match():
    truth = [issue("underage", ["date_of_birth"])]
    pred = [issue("underage", ["date_of_birth"])]
    prf = score_issue_detection(pred, truth)
    assert (prf.tp, prf.fp, prf.fn) == (1, 0, 0)


def test_issue_detection_field_set_must_match():
    truth = [issue("cnic_gender_parity", ["cnic", "gender_male", "gender_female"])]
    pred = [issue("cnic_gender_parity", ["cnic"])]          # same rule, fewer fields: not a match
    prf = score_issue_detection(pred, truth)
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_issue_detection_extra_predicted_issue_is_a_false_positive():
    truth = [issue("underage", ["date_of_birth"])]
    pred = [issue("underage", ["date_of_birth"]), issue("name_mismatch", ["full_name", "declaration_name"])]
    prf = score_issue_detection(pred, truth)
    assert (prf.tp, prf.fp, prf.fn) == (1, 1, 0)


def test_issue_detection_filtered_by_type():
    truth = [issue("underage", ["date_of_birth"], "inconsistency"), issue("invalid_cnic", ["cnic"], "format")]
    pred = [issue("underage", ["date_of_birth"], "inconsistency")]
    prf = score_issue_detection(pred, truth, types=["inconsistency"])
    assert (prf.tp, prf.fp, prf.fn) == (1, 0, 0)
    prf_fmt = score_issue_detection(pred, truth, types=["format"])
    assert (prf_fmt.tp, prf_fmt.fp, prf_fmt.fn) == (0, 0, 1)


# ---------- missing vs unreadable confusion ----------
def test_confusion_matrix_perfect():
    truth = [fr("a", status=S.MISSING), fr("b", status=S.UNREADABLE)]
    pred = [fr("a", status=S.MISSING), fr("b", status=S.UNREADABLE)]
    cm = score_missing_vs_unreadable(pred, truth)
    assert cm.get("missing", "missing") == 1
    assert cm.get("unreadable", "unreadable") == 1


def test_confusion_matrix_confusion_and_other():
    truth = [fr("a", status=S.MISSING), fr("b", status=S.UNREADABLE)]
    pred = [fr("a", status=S.UNREADABLE), fr("b", "readable text", status=S.OK)]
    cm = score_missing_vs_unreadable(pred, truth)
    assert cm.get("missing", "unreadable") == 1
    assert cm.get("unreadable", "other") == 1


def test_confusion_matrix_ignores_ok_and_invalid_ground_truth():
    truth = [fr("a", "fine", status=S.OK), fr("b", status=S.MISSING)]
    pred = [fr("a", "fine", status=S.OK), fr("b", status=S.MISSING)]
    cm = score_missing_vs_unreadable(pred, truth)
    assert sum(cm.counts.values()) == 1
