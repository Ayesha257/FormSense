"""Headless smoke tests of both Streamlit apps (streamlit.testing). No OCR models, no network."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

pytest.importorskip("streamlit")
pytest.importorskip("langchain_community")
pytest.importorskip("faiss")

import streamlit as st  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import formsense.rag as rag_pkg  # noqa: E402
from formsense import rating as R  # noqa: E402
from formsense.rag import GuidelineRetriever  # noqa: E402

MAIN = str(BACKEND / "app" / "streamlit_app.py")
RATE = str(BACKEND / "app" / "rate_explanations.py")


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    r = subprocess.run([sys.executable, "-m", "formsense.generator.generate", "--count", "6", "--split", "dev",
                        "--seed", "11", "--out", str(out)], cwd=BACKEND, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture
def main_app(monkeypatch, sample_dir, tmp_path):
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setenv("FORMSENSE_SAMPLE_DIR", str(sample_dir))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    # fake embeddings must never share the real index folder (their vectors have another size)
    keyword = lambda *a, **k: GuidelineRetriever(index_dir=tmp_path / "index",  # noqa: E731
                                                 embeddings=DeterministicFakeEmbedding(size=32), mode="tfidf")
    monkeypatch.setattr(rag_pkg, "GuidelineRetriever", keyword)
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    assert not at.exception
    return at


def open_sample(at, n=2):
    at.sidebar.checkbox[0].uncheck()            # Run OCR off: no model download in tests
    at.selectbox[0].select_index(n).run()
    assert not at.exception, at.exception
    return at


def test_main_app_starts_and_lists_samples(main_app):
    assert main_app.selectbox[0].label.startswith("...or try a sample")
    assert len(main_app.selectbox[0].options) == 1 + 6


def test_main_app_shows_every_tab_metrics_and_downloads(main_app):
    at = open_sample(main_app)
    assert [t.label for t in at.tabs] == ["Overview", "Issues", "Checklist", "Fields", "Guidelines"]
    assert [m.label for m in at.metric[:5]] == ["Items OK", "Missing", "Unreadable", "Invalid", "Issues"]
    assert any("OCR is off" in i.value for i in at.info)
    assert not at.error


def test_issue_highlight_selector_runs(main_app):
    at = open_sample(main_app)
    box = next(s for s in at.selectbox if s.key == "focus")
    assert len(box.options) > 1
    box.select_index(1).run()
    assert not at.exception


def test_field_inspector_selector_runs(main_app):
    at = open_sample(main_app)
    box = next(s for s in at.selectbox if s.key == "inspect")
    box.select_index(3).run()
    assert not at.exception


def test_llm_explanation_button_appears_with_an_explainer(monkeypatch, main_app):
    monkeypatch.setattr("formsense.summary.find_llm_explainer", lambda: (lambda report: "Fix your signature, please."))
    at = open_sample(main_app)
    btn = next(b for b in at.button if b.key == "explain")
    btn.click().run()
    assert not at.exception and not at.error
    assert any("Fix your signature" in m.value for m in at.markdown) or any("Fix your signature" in t.value for t in at.text)


def test_failing_explainer_shows_an_error_not_a_crash(monkeypatch, main_app):
    def boom(report):
        raise RuntimeError("Groq is down")
    monkeypatch.setattr("formsense.summary.find_llm_explainer", lambda: boom)
    at = open_sample(main_app)
    next(b for b in at.button if b.key == "explain").click().run()
    assert not at.exception and any("Groq is down" in e.value for e in at.error)


def test_vlm_compare_is_disabled_without_a_key(main_app):
    vlm_box = next(c for c in main_app.sidebar.checkbox if "vision-language" in c.label)
    assert vlm_box.disabled


def test_vlm_compare_tab_with_a_fake_model(monkeypatch, main_app):
    from test_vlm import good_reply
    from formsense.vlm import parse_response
    import formsense.vlm as vlm_pkg

    class FakeVLM:
        def __init__(self, template, model=None, **kw):
            self.template = template

        def run(self, page):
            return SimpleNamespace(fields=parse_response(json.dumps(good_reply()), self.template)[0])

    monkeypatch.setattr(vlm_pkg, "VLMAuditor", FakeVLM)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    at.sidebar.checkbox[0].uncheck()
    next(c for c in at.sidebar.checkbox if "vision-language" in c.label).check()
    at.selectbox[0].select_index(2).run()
    assert not at.exception, at.exception
    assert "Compare" in [t.label for t in at.tabs]
    assert not at.error


def test_vlm_failure_is_reported_not_raised(monkeypatch, main_app):
    import formsense.vlm as vlm_pkg
    from formsense.vlm import VLMError

    class BrokenVLM:
        def __init__(self, *a, **k):
            pass

        def run(self, page):
            raise VLMError("rate limit exceeded")

    monkeypatch.setattr(vlm_pkg, "VLMAuditor", BrokenVLM)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    at.sidebar.checkbox[0].uncheck()
    next(c for c in at.sidebar.checkbox if "vision-language" in c.label).check()
    at.selectbox[0].select_index(2).run()
    assert not at.exception and any("rate limit exceeded" in e.value for e in at.error)


# ---------------- rating app ----------------

@pytest.fixture
def rating_folder(tmp_path):
    items = [{"id": f"item_{n:03d}", "form_id": f"f{n // 2}", "explanation": f"Explanation {n}",
              "issues": [{"id": "r", "type": "missing", "severity": "error", "fields": ["signature"],
                          "message": "Signature is missing"}], "facts": ["Full Name: Ali"], "image": None}
             for n in range(1, 5)]
    key = {i["id"]: ("llm_only" if n % 2 else "llm_rag") for n, i in enumerate(items)}
    R.save_set(tmp_path / "rating_set.json", items, key, tmp_path / "rating_key.json")
    return tmp_path


def start_rating(folder, name="Tester"):
    at = AppTest.from_file(RATE, default_timeout=30).run()
    at.text_input[0].set_value(name)
    at.text_input[1].set_value(str(folder))
    return at.run()


def test_rating_app_without_a_set_explains_how_to_make_one(tmp_path):
    at = AppTest.from_file(RATE, default_timeout=30).run()
    at.text_input[0].set_value("x")
    at.text_input[1].set_value(str(tmp_path))
    at.run()
    assert any("No rating set" in w.value for w in at.warning)


def test_rating_app_asks_for_a_name_first(rating_folder):
    at = AppTest.from_file(RATE, default_timeout=30).run()
    at.text_input[1].set_value(str(rating_folder))
    at.run()
    assert any("Enter your name" in i.value for i in at.info)


def test_rating_app_never_shows_the_condition(rating_folder):
    at = start_rating(rating_folder)
    shown = " ".join(m.value for m in at.markdown) + " ".join(t.value for t in at.text)
    for leak in ("llm_only", "llm_rag", "LLM alone", "LLM + RAG"):
        assert leak not in shown
    assert at.subheader[0].value == "Item 1 of 4"


def test_rating_app_requires_all_three_scores(rating_folder):
    at = start_rating(rating_folder)
    at.radio[0].set_value(4)
    at.button[-1].click().run()
    assert any("all three" in e.value for e in at.error)
    assert not R.ratings_path("Tester", rating_folder).exists()


def test_rating_app_saves_and_moves_on_and_resumes(rating_folder):
    at = start_rating(rating_folder)
    for r in at.radio:
        r.set_value(5)
    at.text_area[0].set_value("clear")
    at.button[-1].click().run()
    assert not at.exception and at.subheader[0].value.startswith("Item 2 of 4")
    saved = json.loads(R.ratings_path("Tester", rating_folder).read_text())
    assert saved["ratings"]["item_001"]["faithfulness"] == 5 and saved["ratings"]["item_001"]["comment"] == "clear"
    again = start_rating(rating_folder)                      # a new session resumes at the first unrated item
    assert again.subheader[0].value.startswith("Item 2 of 4")


def test_rating_app_finishes(rating_folder):
    at = start_rating(rating_folder)
    for _ in range(4):
        for r in at.radio:
            r.set_value(3)
        at.button[-1].click().run()
    assert any("All items are rated" in s.value for s in at.success)
    assert R.progress(R.load_items(rating_folder / "rating_set.json"), R.load_ratings(R.ratings_path("Tester", rating_folder))) == (4, 4)
