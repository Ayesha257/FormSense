"""Builds the prompt sent to the LLM.

Kept separate from explainer.py so the wording can be tuned, A/B tested, or swapped for a
different language without touching the calling code or the tests around it.
"""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

from formsense.schema import FieldResult, FieldStatus, Issue

SYSTEM_PROMPT = """You are explaining the result of an automated admission form check to the \
applicant who filled it in. You did not design the rules and you cannot change them; you only \
explain, in plain, friendly English, what was found.

Rules:
- Never invent a problem that is not in the issues list, and never invent a value that is not \
given to you.
- Never repeat a CNIC, phone number or email address, even a masked one; refer to the field by \
its label instead (for example "your CNIC" not "35XXX-XXXXXXX-X").
- Do not mention rule ids, field names from code (like "cnic_gender_parity"), JSON, or anything \
about how the system works internally.
- Be concrete about WHAT is wrong, not just that something is wrong: name which field, in plain \
words, for every issue.
- If there are no issues, say so plainly and do not invent cautions.
- Keep the tone like a helpful admissions office clerk, not a legal notice."""


def _field_line(f: FieldResult, label: str) -> str:
    if f.status == FieldStatus.MISSING:
        return f"- {label}: left empty"
    if f.status == FieldStatus.UNREADABLE:
        return f"- {label}: present but not clearly readable"
    shown = f.value or f.raw_text or "(empty)"
    return f"- {label}: {shown}"


def _issue_line(i: Issue, labels: Mapping[str, str]) -> str:
    names = ", ".join(labels.get(n, n) for n in i.fields)
    return f"- issue_id={i.id} | severity={i.severity.value} | fields=[{names}] | detail: {i.message}"


def build_user_prompt(
    fields: Sequence[FieldResult],
    issues: Sequence[Issue],
    labels: Mapping[str, str],
    guideline_context: Optional[Sequence[str]] = None,
) -> str:
    """fields and issues must already have PII masked by the caller."""
    field_lines = "\n".join(_field_line(f, labels.get(f.name, f.name)) for f in fields if f.status != FieldStatus.OK
                             or (f.value or f.raw_text))
    issue_lines = "\n".join(_issue_line(i, labels) for i in issues) or "(none: the form has no issues)"

    parts: List[str] = [
        "Here is what the automated check found on this admission form.",
        "",
        "Fields with something notable (blank or unreadable fields, and any field that has a "
        "problem below):",
        field_lines or "(every field looks filled in)",
        "",
        "Issues found, in the system's own words (translate these into plain English, do not "
        "copy them verbatim):",
        issue_lines,
    ]
    if guideline_context:
        parts += ["", "Relevant excerpts from the official form filling guidelines, for context "
                       "only (cite them only if they explain why something is required):",
                  "\n".join(f"- {c}" for c in guideline_context)]
    parts += [
        "",
        "Write the summary, then one short explanation per issue above (same issue_id, same "
        "order), then one recommendation sentence (or null if there are no issues).",
    ]
    return "\n".join(parts)
