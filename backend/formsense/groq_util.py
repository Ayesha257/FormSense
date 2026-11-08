"""Shared Groq plumbing: current model names, retries, and clear errors for retired models.

Groq retires models regularly (Llama 4 Scout on 2026-07-17, Llama 3.3 70B on 2026-08-16), and a
retired id answers with a 404 "model_not_found". Model ids therefore live in ONE place here and
can always be overridden from backend/.env, so the next retirement is a config change, not a code change:

    GROQ_VLM_MODEL=...    vision model for the VLM comparison
    GROQ_TEXT_MODEL=...   text model for the rating-set explainer
    GROQ_VLM_REASONING=none|low|medium|high    thinking depth ('' = do not send the parameter)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# Groq docs (console.groq.com/docs/vision), checked 2026-11-07.
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"      # vision + JSON mode; "preview" status, so keep it overridable
DEFAULT_TEXT_MODEL = "qwen/qwen3.8-27b"
KNOWN_VISION_MODELS = ("qwen/qwen3.8-27b", "meta-models/muse-spark-open")   # muse-spark-open: limited preview
RETIRED_MODELS = ("meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct",
                  "llama-3.3-70b-versatile", "qwen/qwen3-32b")


class GroqError(RuntimeError):
    """A Groq request failed for good (after retries) or the model is gone."""


def load_env() -> None:
    """Read backend/.env (GROQ_API_KEY and the model overrides) if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except ImportError:
        pass


def vision_model(explicit: str | None = None) -> str:
    load_env()
    return explicit or os.getenv("GROQ_VLM_MODEL") or DEFAULT_VISION_MODEL


def text_model(explicit: str | None = None) -> str:
    load_env()
    return explicit or os.getenv("GROQ_TEXT_MODEL") or DEFAULT_TEXT_MODEL


def make_client():
    """A Groq client from GROQ_API_KEY (our own retries, so the SDK's are off)."""
    load_env()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise GroqError("GROQ_API_KEY is not set. Put it in backend/.env "
                        "(get one at https://console.groq.com/keys).")
    from groq import Groq
    return Groq(api_key=key, max_retries=0)


# ---------------- error classification ----------------

_GONE_WORDS = ("model_not_found", "model_decommissioned", "does not exist", "decommissioned",
               "no longer supported", "has been deprecated")


def is_model_gone(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) not in (400, 404):
        return False
    msg = str(exc).lower()
    return any(w in msg for w in _GONE_WORDS)


def is_rate_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "RateLimitError" or getattr(exc, "status_code", None) == 429


def retry_after(exc: Exception, default: float) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return max(default, float(headers.get("retry-after", 0)))
    except (TypeError, ValueError):
        return default


def available_models(client) -> list[str]:
    """Model ids this key can use (empty list if the lookup itself fails)."""
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return []


def model_gone_message(client, model: str, env_var: str, default: str) -> str:
    msg = (f"Groq does not serve the model '{model}' any more (retired, renamed, or not enabled for your key). "
           f"Set {env_var} in backend/.env to a current model, for example {default}. "
           "Run `python scripts/groq_models.py` to see what your key can use; "
           "Groq's list is at https://console.groq.com/docs/models.")
    ids = available_models(client)
    if ids:
        msg += " Available to your key: " + ", ".join(ids[:15]) + (" ..." if len(ids) > 15 else "") + "."
    return msg


# ---------------- the call ----------------

def chat(client, *, error: type[Exception] = GroqError, env_var: str = "GROQ_VLM_MODEL",
         default_model: str = DEFAULT_VISION_MODEL, reasoning_effort: str | None = None,
         max_retries: int = 4, sleep: Callable[[float], None] = time.sleep, **kwargs):
    """client.chat.completions.create(**kwargs) with retries and friendly errors.

    - 429 and 5xx/network errors are retried with exponential backoff (honouring Retry-After)
    - a retired model raises `error` immediately with what to do about it
    - reasoning_effort is sent as an extra body field; if the model rejects it (400 mentioning
      "reasoning"), the call is repeated without it, so any model id can be used
    - other client errors (bad image, bad key) are not retried
    """
    delay, attempt, last = 4.0, 0, None
    use_reasoning = bool(reasoning_effort)
    model = kwargs.get("model", "")
    while True:
        call = dict(kwargs)
        if use_reasoning:
            call["extra_body"] = {"reasoning_effort": reasoning_effort}
        try:
            return client.chat.completions.create(**call)
        except Exception as e:
            last = e
            if is_model_gone(e):
                raise error(model_gone_message(client, model, env_var, default_model)) from e
            status = getattr(e, "status_code", None)
            if use_reasoning and status == 400 and "reasoning" in str(e).lower():
                use_reasoning = False
                continue
            retryable = is_rate_limit(e) or status is None or (isinstance(status, int) and status >= 500)
            if not retryable or attempt >= max_retries:
                break
            sleep(retry_after(e, delay))
            delay = min(delay * 2, 60.0)
            attempt += 1
    raise error(f"Groq request failed: {last}") from last
