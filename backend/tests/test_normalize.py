import pytest

from formsense.nlp.normalize import (
    names_match, normalize_cnic, normalize_date, normalize_email, normalize_name, normalize_number,
    normalize_phone, normalize_year, parse_date_strict, pick_format, to_number,
)


# ---------- strict by default: real applicant mistakes must stay visible ----------
@pytest.mark.parametrize("raw", ["3520212345671", "35202-123456-1", "35202-1234567-12", "35202-12345O7-1", "N/A", ""])
def test_cnic_errors_are_not_hidden(raw):
    assert not normalize_cnic(raw).ok


def test_cnic_valid():
    n = normalize_cnic(" 35202-1234567-1 ")
    assert n.ok and n.value == "35202-1234567-1" and not n.repaired


# ---------- repair only on request, and only when it is unambiguous ----------
def test_cnic_repair_fixes_ocr_confusions_and_dashes():
    assert normalize_cnic("35202-12345O7-1", repair=True).value == "35202-1234507-1"
    assert normalize_cnic("3520212345671", repair=True).value == "35202-1234567-1"
    assert normalize_cnic("35202 1234567 1", repair=True).repaired


def test_cnic_repair_does_not_invent_digits():
    assert not normalize_cnic("35202-12345678-1", repair=True).ok      # 14 digits
    assert not normalize_cnic("35202-123456-1", repair=True).ok        # 12 digits


def test_phone():
    assert normalize_phone("0300-1234567").ok
    assert not normalize_phone("03001234567").ok
    assert normalize_phone("+92 300 1234567", repair=True).value == "0300-1234567"
    assert normalize_phone("0092-300-1234567", repair=True).value == "0300-1234567"
    assert not normalize_phone("0400-1234567", repair=True).ok


def test_email():
    assert normalize_email("a.b@example.com").ok
    assert not normalize_email("a.b@example").ok
    assert not normalize_email("a b @ example . com").ok
    assert normalize_email("ali (at) gmail,com", repair=True).value == "ali@gmail.com"


# ---------- dates ----------
@pytest.mark.parametrize("raw,iso", [
    ("14/03/2005", "2005-03-14"), ("14-03-2005", "2005-03-14"), ("14 Mar 2005", "2005-03-14"),
    ("14.03.2005", "2005-03-14"), ("2005-03-14", "2005-03-14"), ("5th March 2005", "2005-03-05"),
])
def test_dates_parse_to_iso(raw, iso):
    n = normalize_date(raw)
    assert n.ok and n.value == iso


@pytest.mark.parametrize("raw", ["31/02/2005", "45/13/2004", "00/00/0000", "soon", "N/A", "", "March 2005"])
def test_impossible_or_partial_dates_fail(raw):
    assert not normalize_date(raw).ok


def test_date_repair_only_when_asked():
    assert not normalize_date("O5/O3/2OO5").ok
    n = normalize_date("O5/O3/2OO5", repair=True)
    assert n.ok and n.value == "2005-03-05" and n.repaired


def test_dateparser_can_be_disabled():
    assert not normalize_date("5th March 2005", use_dateparser=False).ok


def test_parse_date_strict_is_deterministic_dmy():
    assert parse_date_strict("03/04/2005").month == 4       # day first


# ---------- numbers ----------
@pytest.mark.parametrize("raw,val", [("85", "85"), ("85%", "85"), (" 85.5 % ", "85.5"), ("1,050", "1050")])
def test_numbers(raw, val):
    assert normalize_number(raw).value == val


@pytest.mark.parametrize("raw", ["eighty", "-5", "nan", "inf", "1e3", "", "8O"])
def test_non_numbers_fail(raw):
    assert not normalize_number(raw).ok


def test_number_repair():
    assert normalize_number("8O", repair=True).value == "80"
    assert not normalize_number("BIDS", repair=True).ok          # no real digit: never invent a number


def test_to_number():
    assert to_number("85%") == 85.0 and to_number("abc") is None


def test_year():
    assert normalize_year("2019").ok and not normalize_year("20x9").ok and not normalize_year("19").ok
    assert normalize_year("2O19", repair=True).value == "2019"


# ---------- names ----------
def test_name_cleaning():
    assert normalize_name("  ali   RAZA. ").value == "ali RAZA"
    assert not normalize_name("---").ok


def test_name_match_is_order_and_case_insensitive():
    assert names_match("Ali Raza", "raza, ali")
    assert names_match("Muhammad Ali Khan", "Muhammad Ali  Khan.")
    assert not names_match("Ali Raza", "Zainab Bukhari")


def test_pick_format():
    assert pick_format("cnic", "text", "text") == "cnic"
    assert pick_format("date_of_birth", "date", "date") == "date"
    assert pick_format("matric_total", "text", "number") == "number"
    assert pick_format("matric_year", "text", "text") == "year"
    assert pick_format("father_name", "text", "text") == "name"
    assert pick_format("address", "text", "text") == "text"
