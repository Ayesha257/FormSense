"""Tests for formsense/llm/pii_mask.py."""
from __future__ import annotations

import pytest

from formsense.llm.pii_mask import (
    ALWAYS_MASKED, assert_no_pii, mask_cnic, mask_email, mask_fields, mask_phone, mask_text, mask_value,
)
from formsense.schema import FieldResult, FieldStatus as S


def test_always_masked_set():
    assert ALWAYS_MASKED == {"cnic", "phone", "email"}


@pytest.mark.parametrize("raw,expected", [
    ("35202-1234567-1", "35XXX-XXXXXXX-X"),
    ("not-a-cnic", "[REDACTED CNIC]"),
])
def test_mask_cnic(raw, expected):
    assert mask_cnic(raw) == expected


@pytest.mark.parametrize("raw,expected", [("0300-1234567", "03XX-XXXXXXX"), ("garbage", "[REDACTED PHONE]")])
def test_mask_phone(raw, expected):
    assert mask_phone(raw) == expected


@pytest.mark.parametrize("raw,expected", [("ali@example.com", "a***@example.com"), ("not-an-email", "[REDACTED EMAIL]")])
def test_mask_email(raw, expected):
    assert mask_email(raw) == expected


def test_mask_value_dispatches_by_field_name():
    assert mask_value("cnic", "35202-1234567-1") == "35XXX-XXXXXXX-X"
    assert mask_value("full_name", "Ali Raza") == "Ali Raza"   # unknown field passes through


def fr(name, raw, status=S.OK):
    return FieldResult(name=name, bbox=(0, 0, 1, 1), raw_text=raw, value=raw or None, status=status)


def test_mask_fields_always_masks_cnic_phone_email_regardless_of_config():
    fields = [fr("cnic", "35202-1234567-1"), fr("phone", "0300-1234567"), fr("email", "a@b.com")]
    out = mask_fields(fields, extra_fields=())      # caller passed nothing extra
    assert all(f.raw_text != orig.raw_text for f, orig in zip(out.fields, fields))


def test_mask_fields_leaves_other_fields_untouched():
    fields = [fr("full_name", "Ali Raza"), fr("address", "House 1, Lahore")]
    out = mask_fields(fields)
    assert [f.raw_text for f in out.fields] == ["Ali Raza", "House 1, Lahore"]


def test_mask_fields_skips_missing_and_unreadable():
    fields = [fr("cnic", "", status=S.MISSING)]
    out = mask_fields(fields)
    assert out.fields[0].raw_text == "" and "cnic" not in out.unmasked_by_name


def test_mask_fields_can_mask_extra_fields():
    fields = [fr("address", "House 1, Lahore")]
    out = mask_fields(fields, extra_fields=["address"])
    assert out.fields[0].raw_text != "House 1, Lahore"


def test_unmasked_map_holds_the_real_values():
    fields = [fr("cnic", "35202-1234567-1")]
    out = mask_fields(fields)
    assert out.unmasked_by_name == {"cnic": "35202-1234567-1"}


def test_mask_text_replaces_real_values_found_in_free_text():
    fields = [fr("cnic", "35202-1234567-1"), fr("phone", "0300-1234567")]
    text = "Called 0300-1234567 about CNIC 35202-1234567-1."
    out = mask_text(text, fields)
    assert "0300-1234567" not in out and "35202-1234567-1" not in out


def test_assert_no_pii_raises_on_leak_and_passes_on_clean_text():
    fields = [fr("cnic", "35202-1234567-1")]
    with pytest.raises(ValueError):
        assert_no_pii("CNIC is 35202-1234567-1", fields)
    assert_no_pii("CNIC is 35XXX-XXXXXXX-X", fields)   # no exception


def test_assert_no_pii_ignores_fields_not_in_scope():
    fields = [fr("full_name", "Ali Raza")]
    assert_no_pii("The applicant is named Ali Raza", fields)   # names are not in ALWAYS_MASKED
