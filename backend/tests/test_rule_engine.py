"""Tests for the YAML rule engine (Week 4)."""
from __future__ import annotations

import datetime as dt
import logging
import random
import warnings

import pytest
import yaml

from conftest import BACKEND, discover_generator_today
from form_factory import TODAY, clean_form, corrupted_form, _fr
from formsense.nlp.template_info import TemplateInfo
from formsense.rules import reference_validator as ref
from formsense.rules.codes import COARSE, SEVERITY, Code
from formsense.rules import engine as engine_mod
from formsense.rules.engine import DEFAULT_RULES_PATH, RuleConfigError, RuleEngine, issue_key, load_rules, mark_invalid
from formsense.schema import FieldResult, FieldStatus as S, IssueType


def ids(issues):
    return sorted(i.id for i in issues)


def form(tpl, seed=0):
    return clean_form(tpl, random.Random(seed))


def issues_for(engine, tpl, forms):
    return engine.validate(list(forms.values()), TODAY)


# =============================================================== the YAML matches codes.py
def test_yaml_ids_types_and_severities_match_codes_py():
    rules, _ = load_rules()
    by_id = {r.id: r for r in rules}
    assert set(by_id) == {c.value for c in Code}, "rules.yaml and codes.py list different rules"
    for code in Code:
        r = by_id[code.value]
        assert r.type == COARSE[code], f"{code.value}: type differs from codes.py"
        assert r.severity == SEVERITY[code], f"{code.value}: severity differs from codes.py"


def test_every_rule_has_a_description_and_message():
    for r in load_rules()[0]:
        assert r.description and r.message


def test_rule_catalogue(mini_engine):
    assert len(mini_engine.catalogue()) == 19


# =============================================================== clean forms
def test_clean_forms_have_no_issues(mini_engine, mini_tpl):
    for seed in range(100):
        assert issues_for(mini_engine, mini_tpl, form(mini_tpl, seed)) == []


# =============================================================== one test per rule
def put(forms, tpl, name, raw, **kw):
    forms[name] = _fr(tpl.fields[name], raw, **kw)


def only(engine, tpl, forms, rule_id, fields=None):
    got = [i for i in issues_for(engine, tpl, forms) if i.id == rule_id]
    assert len(got) == 1, f"expected exactly one {rule_id}, got {ids(issues_for(engine, tpl, forms))}"
    if fields is not None:
        assert set(got[0].fields) == set(fields)
    return got[0]


def test_missing_required(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "address", "", status=S.MISSING, ink=0.0)
    only(mini_engine, mini_tpl, f, "missing_required", ["address"])


def test_missing_signature(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "signature", "", status=S.MISSING, ink=0.0)
    only(mini_engine, mini_tpl, f, "missing_required", ["signature"])


def test_status_ok_but_no_text_still_counts_as_missing(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "address", "")
    only(mini_engine, mini_tpl, f, "missing_required", ["address"])


def test_unreadable(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "address", "x#~s", status=S.UNREADABLE, conf=0.3)
    i = only(mini_engine, mini_tpl, f, "unreadable_field", ["address"])
    assert i.severity.value == "warning"


@pytest.mark.parametrize("raw", ["N/A", "n/a", "-", "nil", "none", "xxxx", "..."])
def test_placeholder(mini_engine, mini_tpl, raw):
    f = form(mini_tpl); put(f, mini_tpl, "address", raw)
    only(mini_engine, mini_tpl, f, "placeholder_value", ["address"])


@pytest.mark.parametrize("raw", ["3520212345671", "35202-123456-1", "35202-1234567-12", "35202-12345O7-1"])
def test_invalid_cnic(mini_engine, mini_tpl, raw):
    f = form(mini_tpl); put(f, mini_tpl, "cnic", raw)
    only(mini_engine, mini_tpl, f, "invalid_cnic", ["cnic"])


@pytest.mark.parametrize("raw", ["a@b", "name.example.com", "a@@b.com"])
def test_invalid_email(mini_engine, mini_tpl, raw):
    f = form(mini_tpl); put(f, mini_tpl, "email", raw)
    only(mini_engine, mini_tpl, f, "invalid_email", ["email"])


@pytest.mark.parametrize("raw", ["0400-1234567", "03001234567", "0300-123456"])
def test_invalid_phone(mini_engine, mini_tpl, raw):
    f = form(mini_tpl); put(f, mini_tpl, "phone", raw)
    only(mini_engine, mini_tpl, f, "invalid_phone", ["phone"])


@pytest.mark.parametrize("raw", ["31/02/2005", "45/13/2004", "soon"])
def test_invalid_date(mini_engine, mini_tpl, raw):
    f = form(mini_tpl); put(f, mini_tpl, "date_of_birth", raw)
    only(mini_engine, mini_tpl, f, "invalid_date", ["date_of_birth"])


@pytest.mark.parametrize("name,raw", [("matric_total", "eighty"), ("inter_obtained", "8O0"), ("matric_year", "20x9")])
def test_invalid_number(mini_engine, mini_tpl, name, raw):
    f = form(mini_tpl); put(f, mini_tpl, name, raw)
    only(mini_engine, mini_tpl, f, "invalid_number", [name])


def test_no_selection_in_each_group(mini_engine, mini_tpl):
    f = form(mini_tpl)
    for n in mini_tpl.groups["program"]:
        f[n] = _fr(mini_tpl.fields[n], checked=False, ink=0.0)
    only(mini_engine, mini_tpl, f, "no_selection", mini_tpl.groups["program"])


def test_multiple_selections(mini_engine, mini_tpl):
    f = form(mini_tpl)
    for n in mini_tpl.groups["program"][:2]:
        f[n] = _fr(mini_tpl.fields[n], checked=True, ink=0.2)
    only(mini_engine, mini_tpl, f, "multiple_selections", mini_tpl.groups["program"])


def test_underage_boundary(mini_engine, mini_tpl):
    f = form(mini_tpl)
    put(f, mini_tpl, "date_of_birth", "06/11/2010")                   # turns 16 exactly today (TODAY = 2026-11-06)
    assert "underage" not in ids(issues_for(mini_engine, mini_tpl, f))
    put(f, mini_tpl, "date_of_birth", "07/11/2010")                   # turns 16 tomorrow
    only(mini_engine, mini_tpl, f, "underage", ["date_of_birth"])


def test_declaration_in_future(mini_engine, mini_tpl):
    f = form(mini_tpl)
    put(f, mini_tpl, "declaration_date", "06/11/2026")                # today is fine (benign case)
    assert "declaration_date_future" not in ids(issues_for(mini_engine, mini_tpl, f))
    put(f, mini_tpl, "declaration_date", "07/11/2026")
    only(mini_engine, mini_tpl, f, "declaration_date_future", ["declaration_date"])


def test_declaration_before_dob(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "declaration_date", "01/01/2001")
    only(mini_engine, mini_tpl, f, "declaration_before_dob", ["date_of_birth", "declaration_date"])


def test_cnic_gender_parity(mini_engine, mini_tpl):
    f = form(mini_tpl)
    male = f["gender_male"].checked
    f["gender_male"] = _fr(mini_tpl.fields["gender_male"], checked=not male)
    f["gender_female"] = _fr(mini_tpl.fields["gender_female"], checked=male)
    only(mini_engine, mini_tpl, f, "cnic_gender_parity", ["cnic", "gender_male", "gender_female"])


def test_year_order(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "matric_year", "2023"); put(f, mini_tpl, "inter_year", "2021")
    only(mini_engine, mini_tpl, f, "year_order", ["matric_year", "inter_year"])
    put(f, mini_tpl, "inter_year", "2023")                            # equal years are also wrong
    only(mini_engine, mini_tpl, f, "year_order")


def test_marks_exceed_total_and_does_not_double_report(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "inter_obtained", "1500")
    only(mini_engine, mini_tpl, f, "marks_exceed_total", ["inter_total", "inter_obtained"])
    assert "percentage_mismatch" not in ids(issues_for(mini_engine, mini_tpl, f))


def test_percentage_mismatch_and_tolerance(mini_engine, mini_tpl):
    f = form(mini_tpl)
    put(f, mini_tpl, "matric_total", "1000"); put(f, mini_tpl, "matric_obtained", "800")
    put(f, mini_tpl, "matric_percentage", "80.9")                     # within 1 point
    assert "percentage_mismatch" not in ids(issues_for(mini_engine, mini_tpl, f))
    put(f, mini_tpl, "matric_percentage", "50")
    only(mini_engine, mini_tpl, f, "percentage_mismatch", ["matric_total", "matric_obtained", "matric_percentage"])


def test_percentage_with_percent_sign_is_fine(mini_engine, mini_tpl):
    f = form(mini_tpl)
    put(f, mini_tpl, "matric_total", "1000"); put(f, mini_tpl, "matric_obtained", "800")
    put(f, mini_tpl, "matric_percentage", "80%")
    assert issues_for(mini_engine, mini_tpl, f) == []


def test_name_mismatch_but_case_and_spacing_variants_pass(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "full_name", "Ali Raza"); put(f, mini_tpl, "declaration_name", "ALI  RAZA")
    assert issues_for(mini_engine, mini_tpl, f) == []
    put(f, mini_tpl, "declaration_name", "Zainab Bukhari")
    only(mini_engine, mini_tpl, f, "name_mismatch", ["full_name", "declaration_name"])


def test_faint_pencil_signature_is_not_an_error(mini_engine, mini_tpl):
    f = form(mini_tpl); f["signature"] = _fr(mini_tpl.fields["signature"], "", ink=0.012)
    assert issues_for(mini_engine, mini_tpl, f) == []


# =============================================================== cascade control
def test_blank_cnic_does_not_cascade_into_gender_check(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "cnic", "", status=S.MISSING, ink=0.0)
    f["gender_male"] = _fr(mini_tpl.fields["gender_male"], checked=True)
    f["gender_female"] = _fr(mini_tpl.fields["gender_female"], checked=True)   # two ticked, a real group error
    assert ids(issues_for(mini_engine, mini_tpl, f)) == ["missing_required", "multiple_selections"]


def test_bad_dob_hides_age_and_date_order_checks(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "date_of_birth", "31/02/2005")
    assert ids(issues_for(mini_engine, mini_tpl, f)) == ["invalid_date"]


def test_one_issue_per_field_from_field_level_rules(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "cnic", "N/A")               # placeholder AND bad format; report once
    assert ids(issues_for(mini_engine, mini_tpl, f)) == ["placeholder_value"]


def test_empty_optional_field_is_not_a_format_error(mini_tpl):
    raw = {"name": "t", "fields": [
        {"name": "full_name", "label": "Name", "type": "text", "required": True, "bbox": [0, 0, 1, 1]},
        {"name": "email", "label": "Email", "type": "text", "required": False, "bbox": [0, 0, 1, 1]},
    ]}
    eng = RuleEngine(template=raw, raise_errors=True)
    out = eng.validate([FieldResult(name="full_name", bbox=(0, 0, 1, 1), raw_text="Ali", value="Ali"),
                        FieldResult(name="email", bbox=(0, 0, 1, 1), raw_text="", status=S.OK)], TODAY)
    assert out == []


# =============================================================== value versus raw_text
def test_engine_uses_value_when_present_else_raw(mini_engine, mini_tpl):
    f = form(mini_tpl)
    f["date_of_birth"] = f["date_of_birth"].model_copy(update={"raw_text": "5th March 2005", "value": "2005-03-05"})
    assert issues_for(mini_engine, mini_tpl, f) == []                 # ISO value accepted
    f["date_of_birth"] = f["date_of_birth"].model_copy(update={"value": None})
    assert ids(issues_for(mini_engine, mini_tpl, f)) == ["invalid_date"]   # raw is not a known format


# =============================================================== evidence, messages, privacy
def test_issue_carries_evidence_boxes_and_message(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "cnic", "3520212345671")
    i = only(mini_engine, mini_tpl, f, "invalid_cnic")
    assert i.evidence == [tuple(mini_tpl.fields["cnic"].bbox)]
    assert "CNIC" in i.message and i.type == IssueType.FORMAT


def test_messages_never_contain_field_values(mini_engine, mini_tpl):
    rng = random.Random(5)
    secrets = []
    for _ in range(200):
        fields = corrupted_form(mini_tpl, rng, rng.randint(1, 4))
        secrets = [f.raw_text for f in fields if f.name in ("cnic", "phone", "email") and len(f.raw_text) > 5]
        for i in mini_engine.validate(fields, TODAY):
            for s in secrets:
                assert s not in i.message, f"{i.id} leaked a value into its message"


# =============================================================== robustness
def test_absent_template_fields_are_treated_as_missing(mini_engine, mini_tpl):
    f = form(mini_tpl); del f["address"]
    assert ids(issues_for(mini_engine, mini_tpl, f)) == ["missing_required"]


def test_strict_mode_rejects_absent_fields(mini_engine, mini_tpl):
    f = form(mini_tpl); del f["address"]
    with pytest.raises(ValueError):
        mini_engine.validate(list(f.values()), TODAY, strict=True)


def test_unknown_input_fields_are_ignored(mini_engine, mini_tpl):
    f = list(form(mini_tpl).values()) + [FieldResult(name="stray", bbox=(0, 0, 1, 1), raw_text="x")]
    assert mini_engine.validate(f, TODAY) == []


def test_a_crashing_rule_does_not_stop_the_others(mini_tpl, monkeypatch, caplog):
    eng = RuleEngine(template=mini_tpl)                               # raise_errors False (production setting)
    scope, original = engine_mod.CHECKS["min_age"]
    monkeypatch.setitem(engine_mod.CHECKS, "min_age", (scope, lambda ctx, rule: 1 / 0))
    f = form(mini_tpl); put(f, mini_tpl, "cnic", "3520212345671")
    with caplog.at_level(logging.ERROR, logger="formsense.rules"):
        out = eng.validate(list(f.values()), TODAY)
    assert ids(out) == ["invalid_cnic"]
    assert any("underage" in r.getMessage() for r in caplog.records)


def test_correction_mark_is_dormant_until_vision_supplies_a_flag(mini_engine, mini_tpl):
    flds = list(form(mini_tpl).values())
    assert mini_engine.validate(flds, TODAY) == []
    out = mini_engine.validate(flds, TODAY, flags={"address": {"struck_through"}})
    assert ids(out) == ["correction_mark"] and out[0].severity.value == "warning"


def test_correction_mark_does_not_hide_other_problems(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "address", "N/A")
    out = mini_engine.validate(list(f.values()), TODAY, flags={"address": {"struck_through"}})
    assert ids(out) == ["correction_mark", "placeholder_value"]


def test_mark_invalid_sets_status_for_format_issues_only(mini_engine, mini_tpl):
    f = form(mini_tpl); put(f, mini_tpl, "cnic", "3520212345671"); put(f, mini_tpl, "matric_obtained", "1500")
    flds = list(f.values())
    marked = {x.name: x for x in mark_invalid(flds, mini_engine.validate(flds, TODAY))}
    assert marked["cnic"].status == S.INVALID
    assert marked["matric_obtained"].status == S.OK                   # inconsistencies do not poison a field


# =============================================================== rules file validation
def write_rules(tmp_path, rules, settings=None):
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "settings": settings or {}, "rules": rules}))
    return p


BASE = {"id": "x", "check": "required_missing", "type": "missing", "severity": "error", "message": "m"}


def test_rules_file_errors_are_clear(tmp_path):
    with pytest.raises(RuleConfigError, match="duplicate"):
        load_rules(write_rules(tmp_path, [BASE, BASE]))
    with pytest.raises(RuleConfigError, match="unknown check"):
        load_rules(write_rules(tmp_path, [{**BASE, "check": "nope"}]))
    with pytest.raises(RuleConfigError, match="missing 'message'"):
        load_rules(write_rules(tmp_path, [{k: v for k, v in BASE.items() if k != "message"}]))
    with pytest.raises(RuleConfigError):
        load_rules(write_rules(tmp_path, [{**BASE, "severity": "catastrophic"}]))
    with pytest.raises(RuleConfigError):
        load_rules(tmp_path / "does_not_exist.yaml")


def test_disabled_rule_does_not_run(tmp_path, mini_tpl):
    p = write_rules(tmp_path, [{**BASE, "enabled": False}])
    eng = RuleEngine(p, mini_tpl)
    f = form(mini_tpl); put(f, mini_tpl, "address", "", status=S.MISSING, ink=0.0)
    assert eng.validate(list(f.values()), TODAY) == []


def test_changing_a_threshold_in_yaml_changes_behaviour(tmp_path, mini_tpl):
    doc = yaml.safe_load(DEFAULT_RULES_PATH.read_text())
    for r in doc["rules"]:
        if r["id"] == "underage":
            r["params"]["min_age"] = 30
    p = tmp_path / "rules.yaml"; p.write_text(yaml.safe_dump(doc))
    eng = RuleEngine(p, mini_tpl, raise_errors=True)
    out = eng.validate(list(form(mini_tpl).values()), TODAY)
    assert "underage" in ids(out) and "30" in next(i for i in out if i.id == "underage").message


def test_lint_warns_about_rules_that_can_never_fire():
    raw = {"name": "t", "fields": [{"name": "full_name", "label": "Name", "type": "text", "bbox": [0, 0, 1, 1]}]}
    eng = RuleEngine(template=raw)
    assert any("date_of_birth" in w for w in eng.warnings)
    assert any("'gender'" in w for w in eng.warnings)


def test_real_template_has_no_lint_warnings(tpl):
    assert RuleEngine(template=tpl).warnings == [], "rules.yaml refers to fields the template does not have"


# =============================================================== same answers as the reference validator
@pytest.mark.parametrize("seed", range(4))
def test_engine_equals_reference_validator_on_corrupted_forms(template_dict, tpl, engine, seed):
    rng = random.Random(seed)
    for _ in range(400):
        fields = corrupted_form(tpl, rng, rng.randint(0, 4))
        got = {issue_key(i) for i in engine.validate(fields, TODAY)}
        want = {(t.value, i) for t, i in ref.validate(fields, TODAY, template_dict)}
        assert got == want, f"engine and reference disagree: {got ^ want}"


def test_every_rule_is_exercised_by_the_random_forms(tpl, engine):
    rng, seen = random.Random(11), set()
    for _ in range(1500):
        seen |= {i.id for i in engine.validate(corrupted_form(tpl, rng, rng.randint(1, 3)), TODAY)}
    missing = {c.value for c in Code} - seen - {"correction_mark"}      # that one needs a vision flag
    assert not missing, f"rules never triggered: {missing}"


# =============================================================== the generator's own ground truth
LABEL_DIRS = [BACKEND / "data" / "synthetic" / "dev" / "labels", BACKEND / "data" / "synthetic" / "test" / "labels"]
RELATIVE_TO_TODAY = {"underage", "declaration_date_future"}


@pytest.mark.skipif(not any(d.is_dir() for d in LABEL_DIRS), reason="no generated labels under data/synthetic")
def test_engine_reproduces_every_issue_recorded_by_the_generator(engine):
    import json
    skip = {"correction_mark"}                         # needs a vision signal; the reference validator skips it too
    checked, mismatches, no_date = 0, [], 0
    for d in LABEL_DIRS:
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            doc = json.loads(p.read_text(encoding="utf-8"))
            fields = [FieldResult(**x) for x in doc["fields"]]
            if "reference_date" in doc:
                today = dt.date.fromisoformat(doc["reference_date"])
            else:
                today, no_date = dt.date.today(), no_date + 1
            want = {(x["id"], frozenset(x["fields"])) for x in doc["issues"] if x["id"] not in skip}
            got = {k for k in map(issue_key, engine.validate(fields, today)) if k[0] not in skip}
            checked += 1
            if got != want:
                mismatches.append((p.name, sorted(map(str, want - got)), sorted(map(str, got - want))))
    if no_date:
        warnings.warn(f"{no_date} label file(s) had no 'reference_date'; used today's date for those")
    assert checked > 0
    assert not mismatches, f"{len(mismatches)} of {checked} forms differ, first: {mismatches[:3]}"