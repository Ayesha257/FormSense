"""Shared fixtures for the Week 3 and 4 tests (nlp/ and rules/)."""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from formsense.nlp.template_info import DEFAULT_TEMPLATE_PATH, TemplateInfo  # noqa: E402
from formsense.rules.engine import RuleEngine  # noqa: E402

FIXTURE_TEMPLATE = Path(__file__).parent / "fixtures" / "mini_template.json"
BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def template_dict() -> dict:
    """The real template if it exists, otherwise the bundled mini template."""
    path = DEFAULT_TEMPLATE_PATH if DEFAULT_TEMPLATE_PATH.exists() else FIXTURE_TEMPLATE
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def tpl(template_dict) -> TemplateInfo:
    return TemplateInfo.from_dict(template_dict)


@pytest.fixture(scope="session")
def engine(tpl) -> RuleEngine:
    return RuleEngine(template=tpl, raise_errors=True)


@pytest.fixture(scope="session")
def mini_tpl() -> TemplateInfo:
    return TemplateInfo.from_dict(json.loads(FIXTURE_TEMPLATE.read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def mini_engine(mini_tpl) -> RuleEngine:
    return RuleEngine(template=mini_tpl, raise_errors=True)


def discover_generator_today():
    """The reference date the generator used for 'future date' and 'underage' errors.

    Order: env var FORMSENSE_TODAY (YYYY-MM-DD), then a constant in the generator modules.
    Returns None when it cannot be found.
    """
    env = os.environ.get("FORMSENSE_TODAY")
    if env:
        return dt.date.fromisoformat(env)
    for mod in ("data", "errors", "generate", "layout", "render"):
        try:
            m = importlib.import_module(f"formsense.generator.{mod}")
        except Exception:
            continue
        for attr in ("TODAY", "REF_DATE", "REFERENCE_DATE", "TODAY_DATE", "DEFAULT_TODAY", "GEN_TODAY"):
            v = getattr(m, attr, None)
            if isinstance(v, dt.datetime):
                v = v.date()
            if isinstance(v, dt.date):
                return v
    return None
