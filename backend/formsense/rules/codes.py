"""Fine grained rule codes. Issue.id = Code.value (the rule id); Issue.type = the coarse IssueType from schema.py.
Schema stays untouched: we only fill its free text `id` field with these ids."""
from enum import Enum
from formsense.schema import IssueType as I, Severity as S


class Code(str, Enum):
    # field level
    MISSING_REQUIRED = "missing_required"        # empty required field or empty signature box
    UNREADABLE_FIELD = "unreadable_field"
    PLACEHOLDER_VALUE = "placeholder_value"      # "N/A", "-", "xxxx"
    INVALID_CNIC = "invalid_cnic"
    INVALID_EMAIL = "invalid_email"
    INVALID_PHONE = "invalid_phone"
    INVALID_DATE = "invalid_date"
    INVALID_NUMBER = "invalid_number"
    CORRECTION_MARK = "correction_mark"          # struck out and rewritten value (review)
    # selection groups
    NO_SELECTION = "no_selection"                # no box ticked in gender / program
    MULTIPLE_SELECTIONS = "multiple_selections"
    # cross field
    UNDERAGE = "underage"
    DECLARATION_DATE_FUTURE = "declaration_date_future"
    DECLARATION_BEFORE_DOB = "declaration_before_dob"
    CNIC_GENDER_MISMATCH = "cnic_gender_parity"
    YEAR_ORDER = "year_order"
    MARKS_EXCEED_TOTAL = "marks_exceed_total"
    PERCENTAGE_MISMATCH = "percentage_mismatch"
    NAME_MISMATCH = "name_mismatch"


C = Code
COARSE = {
    C.MISSING_REQUIRED: I.MISSING, C.UNREADABLE_FIELD: I.UNREADABLE,
    C.PLACEHOLDER_VALUE: I.FORMAT, C.INVALID_CNIC: I.FORMAT, C.INVALID_EMAIL: I.FORMAT, C.INVALID_PHONE: I.FORMAT,
    C.INVALID_DATE: I.FORMAT, C.INVALID_NUMBER: I.FORMAT, C.CORRECTION_MARK: I.FORMAT,
    C.NO_SELECTION: I.SELECTION, C.MULTIPLE_SELECTIONS: I.SELECTION,
    C.UNDERAGE: I.INCONSISTENCY, C.DECLARATION_DATE_FUTURE: I.INCONSISTENCY, C.DECLARATION_BEFORE_DOB: I.INCONSISTENCY,
    C.CNIC_GENDER_MISMATCH: I.INCONSISTENCY, C.YEAR_ORDER: I.INCONSISTENCY, C.MARKS_EXCEED_TOTAL: I.INCONSISTENCY,
    C.PERCENTAGE_MISMATCH: I.INCONSISTENCY, C.NAME_MISMATCH: I.INCONSISTENCY,
}
SEVERITY = {c: S.ERROR for c in Code}
SEVERITY[C.UNREADABLE_FIELD] = S.WARNING
SEVERITY[C.CORRECTION_MARK] = S.WARNING
