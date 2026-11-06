"""Ground truth must be self consistent: the independent reference validator, run on the GT fields,
must find exactly the issues the injectors recorded (CORRECTION_MARK is image only, so it is excluded)."""
import datetime as dt, collections
import pytest
from formsense.schema import FieldResult
from formsense.rules.codes import Code
from formsense.generator.generate import build_form, ERR_NAMES
from formsense.rules.reference_validator import validate, load_template

TODAY = dt.date(2026, 11, 6)
TPL = load_template()


def forms(n=300, split="dev", seed=1):
    cyc = 0
    for i in range(n):
        gt, _, k = build_form(i, split, seed, TODAY, cyc, capture=False, tpl=TPL, draw=False); cyc += 1 if k else 0
        yield gt


@pytest.fixture(scope="module")
def all_forms():
    return list(forms())


def test_reference_matches_ground_truth(all_forms):
    bad = []
    for gt in all_forms:
        fr = [FieldResult(**f) for f in gt["fields"]]
        got = collections.Counter(validate(fr, TODAY, TPL))
        exp = collections.Counter((Code(i["id"]), frozenset(i["fields"])) for i in gt["issues"] if i["id"] != "correction_mark")
        if got != exp: bad.append((gt["form_id"], gt["injectors"], gt["benign"], dict(got - exp), dict(exp - got)))
    assert not bad, bad[:5]


def test_every_injector_and_issue_type_is_covered(all_forms):
    inj = {n for g in all_forms for n in g["injectors"]}
    types = {i["id"] for g in all_forms for i in g["issues"]}
    assert inj == set(ERR_NAMES), set(ERR_NAMES) - inj
    assert types == {c.value for c in Code}, {c.value for c in Code} - types


def test_clean_forms_exist_and_are_clean(all_forms):
    clean = [g for g in all_forms if not g["issues"]]
    assert len(clean) > 30 and all(f["status"] == "ok" for g in clean for f in g["fields"])


def test_deterministic():
    a = next(iter(forms(1)))["fields"]; b = next(iter(forms(1)))["fields"]
    assert a == b


def test_ink_matches_status():
    """Vision side contract: MISSING fields have (almost) no ink, normal filled fields have clearly more."""
    cyc = 0
    for i in range(25):
        gt, _, k = build_form(i, "dev", 5, TODAY, cyc, capture=False, tpl=TPL); cyc += 1 if k else 0
        for f in gt["fields"]:
            if f["status"] == "missing": assert f["ink_ratio"] < 0.004, (gt["form_id"], f["name"], f["ink_ratio"])
            if f["status"] == "ok" and f["name"] == "signature": assert f["ink_ratio"] > 0.004
