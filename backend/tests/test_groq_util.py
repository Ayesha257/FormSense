"""Shared Groq helper: model resolution, retries, error classification."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense import groq_util as G  # noqa: E402


class Err(Exception):
    def __init__(self, status, msg="boom"):
        super().__init__(msg)
        self.status_code = status


class Client:
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def test_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(G, "load_env", lambda: None)
    for var in ("GROQ_VLM_MODEL", "GROQ_TEXT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert G.vision_model() == G.DEFAULT_VISION_MODEL == "qwen/qwen3.8-27b"
    assert G.text_model() == G.DEFAULT_TEXT_MODEL
    monkeypatch.setenv("GROQ_VLM_MODEL", "from/env")
    assert G.vision_model() == "from/env" and G.vision_model("explicit") == "explicit"


def test_retired_models_are_not_the_default():
    assert G.DEFAULT_VISION_MODEL not in G.RETIRED_MODELS and G.DEFAULT_TEXT_MODEL not in G.RETIRED_MODELS
    assert "llama-3.3-70b-versatile" in G.RETIRED_MODELS


@pytest.mark.parametrize("status,msg,gone", [
    (404, "model_not_found", True), (400, "The model has been decommissioned", True),
    (404, "The model `x` does not exist", True), (400, "invalid image", False),
    (429, "model_not_found", False), (401, "invalid api key", False)])
def test_is_model_gone(status, msg, gone):
    assert G.is_model_gone(Err(status, msg)) is gone


def test_missing_key_message(monkeypatch):
    monkeypatch.setattr(G, "load_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(G.GroqError, match="GROQ_API_KEY"):
        G.make_client()


def test_chat_retries_then_succeeds():
    sleeps = []
    c = Client(Err(429), Err(503), "ok")
    assert G.chat(c, sleep=sleeps.append, model="m", messages=[]) == "ok"
    assert len(c.calls) == 3 and sleeps[1] > sleeps[0]


def test_chat_gives_up_and_wraps_the_cause():
    c = Client(Err(500, "server down"))
    with pytest.raises(G.GroqError, match="Groq request failed: server down"):
        G.chat(c, sleep=lambda s: None, max_retries=1, model="m", messages=[])
    assert len(c.calls) == 2


def test_chat_does_not_retry_client_errors():
    c = Client(Err(401, "invalid api key"))
    with pytest.raises(G.GroqError):
        G.chat(c, sleep=lambda s: pytest.fail("must not sleep"), model="m", messages=[])
    assert len(c.calls) == 1


def test_chat_retired_model_message_names_the_env_var():
    c = Client(Err(404, "model_not_found"))
    with pytest.raises(G.GroqError) as e:
        G.chat(c, env_var="GROQ_TEXT_MODEL", default_model="qwen/qwen3.8-27b", model="gone/model", messages=[])
    assert "gone/model" in str(e.value) and "GROQ_TEXT_MODEL" in str(e.value) and len(c.calls) == 1
