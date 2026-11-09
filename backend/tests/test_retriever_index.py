"""Regression: an index built with one embedding model must never break search with another.

Real bug: fake 32-dimension embeddings (used by tests) wrote the guidelines index; the app's real
384-dimension query then crashed FAISS with `assert d == self.d`.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

pytest.importorskip("langchain_community")
pytest.importorskip("faiss")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from formsense.rag.retriever import GUIDELINES_DIR, GuidelineRetriever  # noqa: E402


@pytest.fixture
def folder(tmp_path):
    shutil.copy(GUIDELINES_DIR / "admission_form_guidelines.md", tmp_path / "guidelines.md")
    return tmp_path


def retriever(folder, size, mode="hybrid"):
    return GuidelineRetriever(folder, embeddings=DeterministicFakeEmbedding(size=size), mode=mode)


def manifest(folder):
    return json.loads((folder / ".index" / "manifest.json").read_text())


def test_search_survives_an_index_built_with_other_vectors(folder):
    retriever(folder, 32).ensure()
    assert manifest(folder)["dim"] == 32
    r = retriever(folder, 48)               # same folder, same text, different embedding size
    assert not r.is_fresh()                  # detected up front ...
    hits = r.search("Can my mother sign the form for me?", k=2)   # ... and it just works
    assert hits and manifest(folder)["dim"] == 48


def test_dimension_guard_works_even_if_the_manifest_lies(folder):
    retriever(folder, 32).ensure()
    m = manifest(folder)
    m["embedding"] = retriever(folder, 48).embedding_id()      # claims to match, vectors say otherwise
    (folder / ".index" / "manifest.json").write_text(json.dumps(m))
    r = retriever(folder, 48)
    assert r.is_fresh()
    assert r.search("ticked two programs", k=1)               # guard rebuilds instead of crashing
    assert manifest(folder)["dim"] == 48


def test_an_old_manifest_without_the_embedding_field_is_rebuilt(folder):
    r = retriever(folder, 32).ensure()
    m = manifest(folder)
    del m["embedding"]
    (folder / ".index" / "manifest.json").write_text(json.dumps(m))
    assert not retriever(folder, 32).is_fresh()


def test_matching_index_is_reused_not_rebuilt(folder):
    retriever(folder, 32).ensure()
    stamp = (folder / ".index" / "index.faiss").stat().st_mtime_ns
    r = retriever(folder, 32)
    assert r.is_fresh()
    r.ensure()
    assert (folder / ".index" / "index.faiss").stat().st_mtime_ns == stamp


def test_damaged_manifest_means_rebuild_not_crash(folder):
    retriever(folder, 32).ensure()
    (folder / ".index" / "manifest.json").write_text("{not json")
    r = retriever(folder, 32)
    assert not r.is_fresh()
    assert r.search("signature", k=1)


def test_offline_rating_set_does_not_touch_the_real_guidelines_index(tmp_path):
    real = GUIDELINES_DIR / ".index" / "manifest.json"
    before = real.read_bytes() if real.exists() else None
    forms = tmp_path / "forms"
    subprocess.run([sys.executable, "-m", "formsense.generator.generate", "--count", "6", "--split", "dev",
                    "--seed", "9", "--out", str(forms)], cwd=BACKEND, capture_output=True, check=True)
    r = subprocess.run([sys.executable, "scripts/make_rating_set.py", "--dir", str(forms), "--limit", "2",
                        "--offline", "--out", str(tmp_path / "rating")], cwd=BACKEND, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (real.read_bytes() if real.exists() else None) == before


def test_default_model_index_is_fresh_before_the_model_loads(folder, monkeypatch):
    """Regression: is_fresh() runs before the default model is loaded, build() records after.
    The two ids must agree, or the real index looks stale on every start and is rebuilt each time."""
    import types
    monkeypatch.setitem(sys.modules, "langchain_huggingface", types.SimpleNamespace(
        HuggingFaceEmbeddings=lambda **kw: DeterministicFakeEmbedding(size=48)))
    GuidelineRetriever(folder).ensure()                        # default path: builds with the (fake) default model
    assert manifest(folder)["embedding"].startswith("hf:") and manifest(folder)["dim"] == 48
    again = GuidelineRetriever(folder)                         # a new process
    assert again.is_fresh()                                    # ...is fresh...
    assert again._embeddings is None                           # ...without having loaded the model
    stamp = (folder / ".index" / "index.faiss").stat().st_mtime_ns
    again.ensure()
    assert (folder / ".index" / "index.faiss").stat().st_mtime_ns == stamp
