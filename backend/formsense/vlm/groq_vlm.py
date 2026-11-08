"""Stretch goal: ask a vision-language model to read the form instead of FormSense.

The raw form photo goes to a VLM on Groq (Qwen 3.8 27B by default; Llama 4 Scout was retired on 2026-07-17) together
with the list of fields to fill. The model answers with JSON, which becomes the
same FieldResult objects as the vision pipeline and the baseline. Everything
downstream (rules, scoring, comparison tables) therefore treats all three
systems identically. This answers "why not just ask a VLM?".

    from formsense.vlm import VLMAuditor
    vlm = VLMAuditor(template)                  # needs GROQ_API_KEY in the environment
    out = vlm.run(page_bgr)                     # out.fields: list[FieldResult]

What a VLM cannot give us, and why FormSense still matters: no per-field
confidence (ocr_confidence stays None), no pixel evidence for signatures and
ticks, answers that can change between runs, and every form leaves the machine.

Groq limits to keep in mind: request size limits for images (we stay under 3.5 MB of base64) and a
requests-per-minute cap on the free tier. The image is shrunk below the size limit
and every answer is cached on disk, so re-running an evaluation costs nothing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from formsense import groq_util
from formsense.groq_util import GroqError
from formsense.schema import FieldResult, FieldStatus, FieldType, FormTemplate

DEFAULT_MODEL = groq_util.DEFAULT_VISION_MODEL   # override with GROQ_VLM_MODEL in .env or model=...
MAX_IMAGE_B64_BYTES = 3_500_000   # Groq rejects base64 images above 4 MB; keep a margin
MAX_SIDE = 1600                   # long side in pixels; enough to read handwriting on a form page
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "outputs" / "vlm_cache"

SYSTEM_PROMPT = (
    "You are a careful data-entry clerk. You read ONE filled-in university admission form "
    "from a photo or scan and report exactly what the applicant wrote or ticked. "
    "Never correct, complete or guess values. Copy handwriting as written, including mistakes. "
    "Reply with a single JSON object and nothing else."
)


class VLMError(GroqError):
    """The VLM could not produce a usable answer (after retries)."""


class VLMParseError(VLMError):
    """The VLM answered, but not with usable JSON (worth asking once more)."""


@dataclass
class VLMOutput:
    fields: list[FieldResult]
    raw: str                                   # the model's reply, for debugging
    from_cache: bool = False
    seconds: float = 0.0
    tokens: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------- image ----------------

def encode_image(page: np.ndarray, max_side: int = MAX_SIDE,
                 max_b64_bytes: int = MAX_IMAGE_B64_BYTES) -> tuple[str, bytes]:
    """BGR/gray page -> (base64 JPEG string, JPEG bytes) small enough for the Groq API."""
    img = page
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    side = max(img.shape[:2])
    if side > max_side:
        s = max_side / side
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    quality = 90
    while True:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise VLMError("Could not encode the page as JPEG.")
        raw = buf.tobytes()
        b64 = base64.b64encode(raw).decode("ascii")
        if len(b64) <= max_b64_bytes:
            return b64, raw
        if quality > 50:
            quality -= 10
        elif min(img.shape[:2]) > 400:
            img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        else:
            raise VLMError("Image is too large for the VLM even after shrinking.")


# ---------------- prompt ----------------

def build_prompt(template: FormTemplate) -> str:
    """Field list + the exact JSON shape wanted. Built from the template, so it never goes stale."""
    lines = []
    for f in template.fields:
        where = f" ({f.section} section)" if f.section else ""
        if f.type == FieldType.CHECKBOX:
            what = f"tick box{' in group ' + repr(f.group) if f.group else ''}"
        elif f.type == FieldType.SIGNATURE:
            what = "signature box"
        elif f.type == FieldType.DATE:
            what = "handwritten date"
        else:
            what = "handwritten text"
        lines.append(f'- "{f.name}": {f.label}{where}, {what}')
    return (
        "Read the filled-in form in the image. Report these fields:\n"
        + "\n".join(lines)
        + "\n\nRules:\n"
          "1. For text and date fields give \"text\" exactly as written and \"legible\": true/false. "
          "If the box is empty give \"text\": null. If ink is there but you cannot read it, "
          "give your best attempt (or null) and \"legible\": false.\n"
          "2. For tick boxes give \"checked\": true only if the box is ticked/marked, else false.\n"
          "3. For the signature box give \"present\": true only if there is a signature or scribble in it.\n"
          "4. Ignore the printed labels and instructions; report only what the applicant added.\n"
          "5. Do not fix typos, wrong dates or inconsistent values. Report them as they are.\n\n"
          "Return JSON of exactly this shape (one entry for every field above):\n"
          '{"fields": {"<field name>": {"text": "...", "legible": true}, '
          '"<tick box name>": {"checked": false}, "signature": {"present": true}}}'
    )


# ---------------- parsing ----------------

def _truthy(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "checked", "ticked", "present", "signed"):
            return True
        if s in ("false", "no", "n", "0", "unchecked", "unticked", "absent", "empty", "none", ""):
            return False
    return None


def _extract_json(text: str) -> dict:
    """The first JSON object in the reply, tolerating ``` fences and chatter around it."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()   # thinking mode
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        if start < 0:
            raise VLMParseError("The VLM reply contains no JSON object.")
        depth, in_str, esc = 0, False, False
        end = -1
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            raise VLMParseError("The VLM reply has unbalanced JSON.")
        try:
            obj = json.loads(t[start:end + 1])
        except json.JSONDecodeError as e:
            raise VLMParseError(f"The VLM reply is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise VLMParseError("The VLM reply JSON is not an object.")
    return obj


def parse_response(text: str, template: FormTemplate) -> tuple[list[FieldResult], list[str]]:
    """Model reply -> (FieldResults in template order, notes about anything odd)."""
    obj = _extract_json(text)
    data = obj.get("fields", obj)
    if isinstance(data, list):    # tolerate [{"name": ..., ...}, ...]
        data = {d.get("name"): d for d in data if isinstance(d, dict)}
    if not isinstance(data, dict):
        raise VLMParseError("The VLM reply has no 'fields' object.")
    notes: list[str] = []
    out: list[FieldResult] = []
    for spec in template.fields:
        entry = data.get(spec.name)
        if entry is None:
            notes.append(f"{spec.name}: not in the reply")
        if not isinstance(entry, dict):   # a bare value instead of an object
            entry = {} if entry is None else {"text": entry, "checked": entry, "present": entry}
        r = FieldResult(name=spec.name, bbox=(0, 0, 0, 0))   # a VLM gives no boxes
        if spec.type == FieldType.CHECKBOX:
            checked = _truthy(entry.get("checked"))
            r.checked = bool(checked)
            r.value = "checked" if r.checked else "unchecked"
        elif spec.type == FieldType.SIGNATURE:
            present = _truthy(entry.get("present", entry.get("signed")))
            r.value = "signed" if present else None
            r.status = FieldStatus.OK if present else FieldStatus.MISSING
        else:
            text = entry.get("text")
            text = "" if text is None else str(text).strip()
            if text.lower() in ("null", "none", "n/a", "empty"):
                text = ""
            legible = _truthy(entry.get("legible"))
            r.raw_text = text
            if not text and legible is not False:
                r.status = FieldStatus.MISSING
            elif legible is False or not text:
                r.status = FieldStatus.UNREADABLE
            else:
                r.status = FieldStatus.OK
        out.append(r)
    return out, notes


# ---------------- the auditor ----------------

class VLMAuditor:
    """Template + page image -> FieldResults, via a Groq vision model."""

    def __init__(self, template: FormTemplate, model: str | None = None, client=None,
                 cache_dir: Path | str | None = DEFAULT_CACHE, max_retries: int = 4,
                 sleep=time.sleep, reasoning: str | None = None):
        self.template = template
        self.model = groq_util.vision_model(model)
        # Qwen 3.x: "none" = instruct mode (no thinking), right for plain transcription
        self.reasoning = os.getenv("GROQ_VLM_REASONING", "none") if reasoning is None else reasoning
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self._sleep = sleep
        self.prompt = build_prompt(template)

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = groq_util.make_client()
            except GroqError as e:
                raise VLMError(str(e)) from e
        return self._client

    # ---- cache ----
    def _cache_path(self, jpeg: bytes) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(jpeg + self.model.encode() + self.prompt.encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def _call(self, b64: str, temperature: float = 0.1) -> tuple[str, int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ]
        resp = groq_util.chat(
            self.client, error=VLMError, env_var="GROQ_VLM_MODEL", default_model=DEFAULT_MODEL,
            reasoning_effort=self.reasoning or None, max_retries=self.max_retries, sleep=self._sleep,
            model=self.model, messages=messages, temperature=temperature,
            max_completion_tokens=4096, response_format={"type": "json_object"})
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", int(getattr(usage, "total_tokens", 0) or 0)

    def run(self, page: np.ndarray) -> VLMOutput:
        b64, jpeg = encode_image(page)
        path = self._cache_path(jpeg)
        t0 = time.time()
        raw, tokens, cached = None, 0, False
        if path is not None and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
                cached = True
            except (OSError, ValueError, KeyError):
                raw = None
        if raw is not None:
            fields, notes = parse_response(raw, self.template)
        else:
            try:
                raw, tokens = self._call(b64)
                fields, notes = parse_response(raw, self.template)
            except VLMParseError:
                # unusable JSON: ask once more, a little less deterministic
                raw, extra = self._call(b64, temperature=0.4)
                tokens += extra
                fields, notes = parse_response(raw, self.template)
        if path is not None and not cached:                  # cache only replies that parsed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "raw": raw}), encoding="utf-8")
        return VLMOutput(fields, raw, cached, round(time.time() - t0, 2), tokens, notes)
