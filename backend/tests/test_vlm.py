"""VLM comparison: parsing, prompt, retries, cache. No network and no API key needed."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formsense.schema import FieldStatus  # noqa: E402
from formsense.vision import load_template  # noqa: E402
from formsense.vlm import VLMAuditor, VLMError, build_prompt, encode_image, parse_response  # noqa: E402

TEMPLATE, _ = load_template()


def good_reply(**over) -> dict:
    fields = {}
    for f in TEMPLATE.fields:
        if f.type.value == "checkbox":
            fields[f.name] = {"checked": f.name in ("gender_male", "program_bscs")}
        elif f.type.value == "signature":
            fields[f.name] = {"present": True}
        else:
            fields[f.name] = {"text": f"text of {f.name}", "legible": True}
    fields.update(over)
    return {"fields": fields}


class FakeClient:
    """Mimics groq.Groq().chat.completions.create with a script of replies or exceptions."""
    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(total_tokens=1234))


class RateLimitError(Exception):
    status_code = 429


class BadRequest(Exception):
    status_code = 400


@pytest.fixture
def page():
    return np.full((900, 700, 3), 255, np.uint8)


def auditor(client, tmp_path, **kw):
    sleeps = []
    a = VLMAuditor(TEMPLATE, client=client, cache_dir=tmp_path / "cache", sleep=sleeps.append, **kw)
    a.sleeps = sleeps
    return a


# ---------------- image ----------------

def test_encode_image_is_small_enough_for_groq():
    noisy = np.random.default_rng(0).integers(0, 255, (4000, 3000, 3), dtype=np.uint8)
    b64, raw = encode_image(noisy)
    assert len(b64) <= 3_500_000
    assert max(cv2.imdecode(np.frombuffer(raw, np.uint8), 1).shape[:2]) <= 1600


def test_encode_image_handles_grayscale_and_small_pages():
    b64, raw = encode_image(np.full((300, 200), 200, np.uint8))
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), 1).shape[:2] == (300, 200)


# ---------------- prompt ----------------

def test_prompt_lists_every_template_field():
    p = build_prompt(TEMPLATE)
    for f in TEMPLATE.fields:
        assert f'"{f.name}"' in p
    assert "legible" in p and "checked" in p and "present" in p


# ---------------- parsing ----------------

def by_name(fields):
    return {r.name: r for r in fields}


def test_parse_good_reply():
    fields, notes = parse_response(json.dumps(good_reply()), TEMPLATE)
    r = by_name(fields)
    assert [x.name for x in fields] == [f.name for f in TEMPLATE.fields]
    assert r["full_name"].raw_text == "text of full_name" and r["full_name"].status == FieldStatus.OK
    assert r["gender_male"].checked is True and r["gender_female"].checked is False
    assert r["signature"].status == FieldStatus.OK and r["signature"].value == "signed"
    assert not notes


def test_parse_empty_and_unreadable_and_unsigned():
    reply = good_reply(full_name={"text": None, "legible": True},
                       address={"text": "sc?ibble", "legible": False},
                       father_name={"text": "", "legible": False},
                       signature={"present": False})
    r = by_name(parse_response(json.dumps(reply), TEMPLATE)[0])
    assert r["full_name"].status == FieldStatus.MISSING
    assert r["address"].status == FieldStatus.UNREADABLE and r["address"].raw_text == "sc?ibble"
    assert r["father_name"].status == FieldStatus.UNREADABLE
    assert r["signature"].status == FieldStatus.MISSING and r["signature"].value is None


def test_parse_tolerates_fences_chatter_and_string_booleans():
    reply = good_reply(gender_male={"checked": "yes"}, gender_female={"checked": "false"})
    text = "Sure! Here is the JSON:\n```json\n" + json.dumps(reply) + "\n```\nHope that helps."
    r = by_name(parse_response(text, TEMPLATE)[0])
    assert r["gender_male"].checked is True and r["gender_female"].checked is False


def test_parse_braces_inside_strings_do_not_break_extraction():
    reply = good_reply(address={"text": "House {12} } Street", "legible": True})
    r = by_name(parse_response("noise " + json.dumps(reply) + " trailing {", TEMPLATE)[0])
    assert r["address"].raw_text == "House {12} } Street"


def test_parse_missing_field_is_noted_and_treated_as_empty():
    reply = good_reply()
    del reply["fields"]["email"], reply["fields"]["program_bsai"]
    fields, notes = parse_response(json.dumps(reply), TEMPLATE)
    r = by_name(fields)
    assert r["email"].status == FieldStatus.MISSING and r["program_bsai"].checked is False
    assert any("email" in n for n in notes)


def test_parse_accepts_flat_and_list_shapes():
    flat = {k: v for k, v in good_reply()["fields"].items()}
    assert by_name(parse_response(json.dumps(flat), TEMPLATE)[0])["cnic"].raw_text == "text of cnic"
    as_list = {"fields": [{"name": k, **v} for k, v in flat.items()]}
    assert by_name(parse_response(json.dumps(as_list), TEMPLATE)[0])["cnic"].raw_text == "text of cnic"


@pytest.mark.parametrize("bad", ["no json here", "{unbalanced", "[1, 2]", '{"fields": 5}'])
def test_parse_rejects_unusable_replies(bad):
    with pytest.raises(VLMError):
        parse_response(bad, TEMPLATE)


def test_vlm_fields_go_through_the_rules():
    import datetime as dt
    from formsense.rules.audit import audit
    reply = good_reply(signature={"present": False}, cnic={"text": "12345-123456-1", "legible": True})
    res = audit(parse_response(json.dumps(reply), TEMPLATE)[0], today=dt.date(2026, 11, 7))
    ids = {i.id for i in res.issues}
    assert any("signature" in i.fields for i in res.issues) and ids   # unsigned form is caught


# ---------------- auditor ----------------

def test_run_calls_groq_with_image_and_json_mode(page, tmp_path):
    client = FakeClient(good_reply())
    out = auditor(client, tmp_path).run(page)
    kw = client.calls[0]
    assert kw["response_format"] == {"type": "json_object"} and kw["temperature"] < 0.3
    parts = kw["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert out.tokens == 1234 and not out.from_cache and len(out.fields) == len(TEMPLATE.fields)


def test_second_run_is_served_from_cache(page, tmp_path):
    client = FakeClient(good_reply())
    a = auditor(client, tmp_path)
    a.run(page)
    again = a.run(page)
    assert again.from_cache and len(client.calls) == 1


def test_cache_is_keyed_on_model(page, tmp_path):
    client = FakeClient(good_reply())
    auditor(client, tmp_path, model="model-a").run(page)
    auditor(client, tmp_path, model="model-b").run(page)
    assert len(client.calls) == 2


def test_rate_limit_is_retried_with_backoff(page, tmp_path):
    client = FakeClient(RateLimitError("slow down"), RateLimitError("slow down"), good_reply())
    a = auditor(client, tmp_path)
    out = a.run(page)
    assert len(client.calls) == 3 and len(a.sleeps) == 2 and a.sleeps[1] > a.sleeps[0]
    assert len(out.fields) == len(TEMPLATE.fields)


def test_client_error_is_not_retried(page, tmp_path):
    client = FakeClient(BadRequest("bad image"))
    a = auditor(client, tmp_path)
    with pytest.raises(VLMError, match="Groq request failed"):
        a.run(page)
    assert len(client.calls) == 1 and not a.sleeps


def test_gives_up_after_max_retries(page, tmp_path):
    client = FakeClient(RateLimitError("x"))
    a = auditor(client, tmp_path, max_retries=2)
    with pytest.raises(VLMError):
        a.run(page)
    assert len(client.calls) == 3


def test_unparseable_reply_is_asked_again_once(page, tmp_path):
    client = FakeClient("I cannot read this image", good_reply())
    out = auditor(client, tmp_path).run(page)
    assert len(client.calls) == 2 and client.calls[1]["temperature"] > 0
    assert len(out.fields) == len(TEMPLATE.fields)


def test_unparseable_twice_fails_and_is_not_cached(page, tmp_path):
    client = FakeClient("nope")
    a = auditor(client, tmp_path)
    with pytest.raises(VLMError):
        a.run(page)
    assert not list((tmp_path / "cache").glob("*.json"))


def test_missing_api_key_gives_a_clear_error(monkeypatch, page, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(VLMError, match="GROQ_API_KEY"):
        VLMAuditor(TEMPLATE, cache_dir=None).run(page)


def test_model_can_be_set_in_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_VLM_MODEL", "some/other-model")
    assert VLMAuditor(TEMPLATE, cache_dir=None).model == "some/other-model"
    monkeypatch.delenv("GROQ_VLM_MODEL")
    assert VLMAuditor(TEMPLATE, cache_dir=None).model == "qwen/qwen3.8-27b"
    assert VLMAuditor(TEMPLATE, model="explicit", cache_dir=None).model == "explicit"


# ---------------- retired models, reasoning mode ----------------

class ModelGone(Exception):
    status_code = 404

    def __init__(self):
        super().__init__("Error code: 404 - {'error': {'message': 'The model `old/model` does not exist or you do "
                         "not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}")


def test_retired_model_gives_an_actionable_error_and_is_not_retried(page, tmp_path):
    client = FakeClient(ModelGone())
    client.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="b-model"), SimpleNamespace(id="a-model")]))
    a = auditor(client, tmp_path, model="old/model")
    with pytest.raises(VLMError) as e:
        a.run(page)
    msg = str(e.value)
    assert "old/model" in msg and "GROQ_VLM_MODEL" in msg and "qwen/qwen3.8-27b" in msg
    assert "a-model, b-model" in msg                      # what the key can use
    assert len(client.calls) == 1 and not a.sleeps


def test_retired_model_error_survives_a_failing_model_list(page, tmp_path):
    client = FakeClient(ModelGone())          # no .models attribute at all
    with pytest.raises(VLMError, match="GROQ_VLM_MODEL"):
        auditor(client, tmp_path).run(page)


def test_instruct_mode_is_requested_by_default(page, tmp_path):
    client = FakeClient(good_reply())
    auditor(client, tmp_path).run(page)
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "none"}
    assert "max_completion_tokens" in client.calls[0]


def test_reasoning_can_be_switched_off(page, tmp_path):
    client = FakeClient(good_reply())
    auditor(client, tmp_path, reasoning="").run(page)
    assert "extra_body" not in client.calls[0]


def test_model_that_rejects_reasoning_is_retried_without_it(page, tmp_path):
    class Unsupported(Exception):
        status_code = 400

    client = FakeClient(Unsupported("reasoning_effort is not supported with this model"), good_reply())
    a = auditor(client, tmp_path)
    out = a.run(page)
    assert "extra_body" in client.calls[0] and "extra_body" not in client.calls[1]
    assert not a.sleeps and len(out.fields) == len(TEMPLATE.fields)


def test_thinking_blocks_in_the_reply_are_ignored():
    text = "<think>The form shows {name: x}. Let me think {</think>\n" + json.dumps(good_reply())
    assert by_name(parse_response(text, TEMPLATE)[0])["cnic"].raw_text == "text of cnic"
