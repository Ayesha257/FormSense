import numpy as np
import pytest

from formsense.nlp.field_mapper import FieldMapper, norm_label


@pytest.fixture(scope="module")
def mapper(tpl):
    return FieldMapper(tpl, use_embeddings=False)


def test_norm_label():
    assert norm_label("D.O.B") == "dob"
    assert norm_label("Father's Name:") == "fathers name"
    assert norm_label("Matric %") == "matric percent"


def test_every_template_label_maps_to_its_own_field(tpl, mapper):
    """Guards against the template and the mapper drifting apart."""
    for name, info in tpl.fields.items():
        r = mapper.map_label(info.label)
        assert r.field == name, f"label {info.label!r} mapped to {r.field} ({r.method}), expected {name}"


@pytest.mark.parametrize("label,field", [
    ("D.O.B", "date_of_birth"), ("Date of Birth:", "date_of_birth"), ("Mobile No.", "phone"),
    ("C.N.I.C", "cnic"), ("E-mail", "email"), ("Fathers Nmae", "father_name"), ("CNlC", "cnic"),
])
def test_known_variants_and_typos(mini_tpl, label, field):
    r = FieldMapper(mini_tpl, use_embeddings=False).map_label(label)
    assert r.field == field


def test_unknown_label_is_not_guessed(mapper):
    r = mapper.map_label("Blood group")
    assert r.field is None and r.method == "none"


def test_male_is_not_confused_with_female(mini_tpl):
    m = FieldMapper(mini_tpl, use_embeddings=False)
    assert m.map_label("Male").field == "gender_male"
    assert m.map_label("Female").field == "gender_female"


def test_one_to_one_assignment_gives_specific_label_the_field(mini_tpl):
    m = FieldMapper(mini_tpl, use_embeddings=False)
    out = m.map_labels_one_to_one(["Date", "Date of Declaration", "D.O.B"])
    assert out[1].field == "declaration_date" and out[2].field == "date_of_birth"
    assert out[0].field is None                       # field already claimed by the more specific label


def _fake_embed(texts):
    axes = {"born": 0, "birth": 0, "father": 1, "guardian": 1, "mobile": 2, "cell": 2}
    out = np.zeros((len(texts), 4))
    for i, t in enumerate(texts):
        for w in t.split():
            if w in axes:
                out[i, axes[w]] += 1
        out[i, 3] = 0.01
    return out


def test_embedding_stage_handles_new_wording(mini_tpl):
    m = FieldMapper(mini_tpl, embed_fn=_fake_embed)
    r = m.map_label("Date you were born")
    assert r.field == "date_of_birth" and r.method == "embedding"


def test_missing_embedding_model_degrades_gracefully(mini_tpl):
    def boom(texts):
        raise RuntimeError("no network")
    m = FieldMapper(mini_tpl, embed_fn=boom)
    assert m.map_label("D.O.B").field == "date_of_birth"          # exact stage still works
    assert m.map_label("some unrelated words").field is None      # embedding stage fails quietly
