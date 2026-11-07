"""Structured shape of the LLM's explanation, used with LangChain's with_structured_output.

Kept separate from formsense/schema.py: that file is the frozen cross-team data contract
(fields, issues, reports). This is an internal detail of how the LLM step talks to the
model and can change without touching anyone else's code.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IssueExplanation(BaseModel):
    issue_id: str = Field(description="The rule id this explanation is for, copied exactly from the input.")
    explanation: str = Field(description="One or two plain English sentences a non technical applicant can "
                                          "understand. No jargon, no field internals, no rule ids in the text.")


class ExplanationOutput(BaseModel):
    summary: str = Field(description="Two to four sentences: does the form look correctly filled overall, "
                                      "and what is the single most important thing to fix, if anything.")
    issues: List[IssueExplanation] = Field(description="One entry per issue given in the input, same order, "
                                                         "same issue_id values. Empty list if there were no issues.")
    overall_recommendation: Optional[str] = Field(
        default=None, description="One sentence of what the applicant should do next, or null if the form has "
                                   "no problems.")
