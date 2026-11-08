"""RAG tests. Fast ones use keyword search + fake embeddings; the slow one loads MiniLM."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("langchain_community")
pytest.importorskip("faiss")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from formsense.rag.retriever import (GUIDELINES_DIR, GuidelineRetriever, expand_query,  # noqa: E402
                                     field_labels, load_chunks, query_for_issue)
from formsense.schema import Issue, IssueType, Severity  # noqa: E402

SAMPLE = GUIDELINES_DIR / "admission_form_guidelines.md"


@pytest.fixture
def folder(tmp_path):
    shutil.copy(SAMPLE, tmp_path / "guidelines.md")
    return tmp_path


def keyword_retriever(folder):
    return GuidelineRetriever(folder, embeddings=DeterministicFakeEmbedding(size=32), mode="tfidf")


def issue(t, fields, msg):
    return Issue(id="x", type=t, severity=Severity.ERROR, fields=fields, message=msg)


def test_chunks_keep_their_section_names(folder):
    docs = load_chunks(folder)
    sections = {d.metadata["section"] for d in docs}
    assert any("Signature of Applicant" in s for s in sections)
    assert any("CNIC Number" in s for s in sections)
    assert not any(d.page_content.lstrip().startswith(">") for d in docs)   # SAMPLE banner skipped


def test_index_rebuilds_when_guidelines_change(folder):
    r = keyword_retriever(folder)
    r.ensure()
    assert r.is_fresh()
    (folder / "guidelines.md").write_text(SAMPLE.read_text(encoding="utf-8") +
                                          "\n## 8. Fees\n\nThe admission fee is non-refundable.\n",
                                          encoding="utf-8")
    assert not r.is_fresh()
    hits = r.search("Is the admission fee refundable?", k=1)
    assert r.is_fresh() and "Fees" in hits[0].section


def test_keyword_search_finds_the_rule(folder):
    r = keyword_retriever(folder)
    assert "Program Applied For" in r.search("ticked more than one program", k=1)[0].section
    assert "Total and Obtained Marks" in r.search("obtained marks greater than total marks", k=1)[0].section


def test_for_issue_returns_citation(folder):
    r = keyword_retriever(folder)
    cites = r.for_issue(issue(IssueType.MISSING, ["signature"], "Signature box is empty"),
                        {"signature": "Signature of Applicant"}, min_score=0.0)
    assert cites and "Signature of Applicant" in cites[0].source and cites[0].issue_id == "x"


def test_query_for_issue_uses_form_labels():
    q = query_for_issue(issue(IssueType.FORMAT, ["cnic"], "CNIC 'cnic' has 11 digits"),
                        {"cnic": "CNIC No."})
    assert "CNIC No." in q and "format" in q and "'cnic'" not in q


def test_heading_boost_prefers_the_fields_own_section(folder):
    r = keyword_retriever(folder)
    labels = field_labels()
    dob = issue(IssueType.FORMAT, ["date_of_birth"], "'31/04/2001' in date_of_birth is not a real calendar date.")
    assert "Date of Birth" in r.search_for_issue(dob, labels, k=1)[0].section
    pct = issue(IssueType.FORMAT, ["inter_percentage"], "'seven hundred' in inter_percentage is not a valid number.")
    assert "Percentage" in r.search_for_issue(pct, labels, k=1)[0].section


def test_field_labels_use_group_names_for_checkboxes():
    labels = field_labels()
    assert labels["program_bsse"] == "Program" and labels["gender_male"] == "Gender"
    assert labels["cnic"] == "CNIC No."


def test_synonyms_are_added():
    assert "Intermediate" in expand_query("My FSc result is late")
    assert "CNIC" in expand_query("my ID card number")
    assert expand_query("hello") == "hello"


@pytest.mark.slow
def test_semantic_retrieval_on_paraphrases(folder):
    pytest.importorskip("sentence_transformers")
    r = GuidelineRetriever(folder)          # hybrid + MiniLM, the real setting
    assert "Signature of Applicant" in r.search("Can my mother sign the form for me?", k=1)[0].section
    assert "Program Applied For" in r.search("I want to apply for both CS and AI", k=1)[0].section
    cites = r.for_issue(issue(IssueType.SELECTION, ["program_bscs", "program_bsse"],
                              "Two programs are ticked"))
    assert cites and "Program Applied For" in cites[0].source
