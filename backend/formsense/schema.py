"""Shared data contract for FormSense.

Every module reads and writes these objects. Vision (Ayesha) produces
FieldResult objects; language/rules (Easha) consume them and produce Issues;
the report bundles everything into an AuditReport.

FROZEN after Week 1: do not change this file without telling the whole team.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# (x, y, width, height) in pixels of the *blank template* image
BBox = tuple[int, int, int, int]


# ---------- Template (what the blank form looks like) ----------

class FieldType(str, Enum):
    TEXT = "text"
    DATE = "date"
    CHECKBOX = "checkbox"        # one tick box
    SIGNATURE = "signature"


class FieldSpec(BaseModel):
    name: str                     # canonical name, e.g. "date_of_birth"
    label: str                    # label printed on the form, e.g. "Date of Birth"
    type: FieldType
    bbox: BBox                    # where the answer is written on the template
    required: bool = True
    group: Optional[str] = None   # checkboxes in the same group, e.g. "program"
    section: Optional[str] = None # e.g. "personal", "academic", "declaration"


class FormTemplate(BaseModel):
    name: str
    image_path: str               # blank template image used for alignment
    width: int
    height: int
    fields: list[FieldSpec]


# ---------- Vision output ----------

class FieldStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"           # no ink in the box
    UNREADABLE = "unreadable"     # ink present but OCR confidence too low
    INVALID = "invalid"           # read fine but fails a rule


class FieldResult(BaseModel):
    name: str
    bbox: BBox
    raw_text: str = ""                     # exactly what OCR returned
    value: Optional[str] = None            # normalised value (e.g. ISO date)
    ocr_confidence: Optional[float] = None # 0..1, None for checkbox/signature
    ink_ratio: float = 0.0                 # share of dark pixels in the box
    checked: Optional[bool] = None         # checkboxes only
    status: FieldStatus = FieldStatus.OK


# ---------- Reasoning output ----------

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueType(str, Enum):
    MISSING = "missing"
    UNREADABLE = "unreadable"
    FORMAT = "format"                 # e.g. bad CNIC pattern
    INCONSISTENCY = "inconsistency"   # cross-field conflict
    SELECTION = "selection"           # e.g. zero or two programs ticked


class Issue(BaseModel):
    id: str                           # rule id, e.g. "cnic_gender_parity"
    type: IssueType
    severity: Severity
    fields: list[str]                 # field names involved
    message: str                      # short technical message
    evidence: list[BBox] = Field(default_factory=list)


class GuidelineCitation(BaseModel):
    issue_id: str
    text: str                         # retrieved guideline passage
    source: str                       # e.g. "guidelines.md#section-3"


class AuditReport(BaseModel):
    form_id: str
    fields: list[FieldResult]
    issues: list[Issue]
    explanation: Optional[str] = None             # LLM-written summary
    citations: list[GuidelineCitation] = Field(default_factory=list)
