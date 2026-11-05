"""FormSense demo UI. Run from the backend folder:

    streamlit run app/streamlit_app.py

Shows the vision step (alignment, field boxes, OCR) and, optionally, the
OCR + regex baseline side by side. The rules, LLM explanation and report
plug in here as those modules are ready.
"""
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from formsense.baseline import BaselineAuditor  # noqa: E402
from formsense.ingest.loader import load_bytes  # noqa: E402
from formsense.schema import FieldResult, FieldStatus, FieldType  # noqa: E402
from formsense.scoring import score_field  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.draw import annotate  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

TEMPLATES = {
    "Admission form": BACKEND / "data" / "templates" / "admission_form.json",
    "Demo form": BACKEND / "data" / "templates" / "demo" / "demo_template.json",
}
REAL_BLANK = BACKEND / "data" / "templates" / "admission_form_blank.png"
DEMO_SAMPLES = BACKEND / "data" / "templates" / "demo" / "samples"

st.set_page_config(page_title="FormSense", layout="wide")


# ---------------- cached models and runs ----------------

@st.cache_resource
def get_template(path: str):
    return load_template(path)


@st.cache_resource
def get_ocr(use_trocr: bool):
    return FieldOCR(use_trocr=use_trocr)


@st.cache_resource
def get_baseline(path: str):
    template, _ = get_template(path)
    return BaselineAuditor(template, reader=get_ocr(False).paddle)


@st.cache_data(show_spinner=False, max_entries=20)
def run_vision(data: bytes, name: str, tpl_path: str, use_ocr: bool, use_trocr: bool):
    template, blank = get_template(tpl_path)
    page = load_bytes(data, name)[0]
    ocr = get_ocr(use_trocr) if use_ocr else None
    out = analyze_page(page, template, blank, ocr=ocr)
    return page, out


@st.cache_data(show_spinner=False, max_entries=20)
def run_baseline(data: bytes, name: str, tpl_path: str):
    page = load_bytes(data, name)[0]
    return get_baseline(tpl_path).run(page)


# ---------------- display helpers ----------------

def shown(r: FieldResult) -> str:
    if r.checked is not None:
        return "☑ ticked" if r.checked else "☐ not ticked"
    if r.status == FieldStatus.MISSING:
        return "— missing"
    if r.status == FieldStatus.UNREADABLE:
        return f"? unreadable ({r.raw_text})" if r.raw_text else "? unreadable"
    return r.raw_text or r.value or "present"


def shown_truth(t: dict) -> str:
    if "checked" in t:
        return "☑ ticked" if t["checked"] else "☐ not ticked"
    if t["status"] == "missing":
        return "— missing"
    return t.get("text") or "present"


def fields_table(results: list[FieldResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Field": r.name, "Status": r.status.value, "Read as": shown(r),
        "OCR confidence": r.ocr_confidence, "Ink": r.ink_ratio,
    } for r in results])


# ---------------- page ----------------

st.title("FormSense: Form Auditor")

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Template", list(TEMPLATES), index=0 if REAL_BLANK.exists() else 1)
    use_ocr = st.checkbox("Run OCR", value=True)
    use_trocr = st.checkbox("Handwriting fallback (TrOCR)", value=True, disabled=not use_ocr)
    compare = st.checkbox("Compare with baseline (OCR + regex)", value=False,
                          help="Runs plain full-page OCR too. Slower: about 15-40 s per form on a laptop CPU.")

tpl_path = str(TEMPLATES[choice])
try:
    template, blank = get_template(tpl_path)
except (FileNotFoundError, ValueError) as e:
    st.warning(f"{choice} is not ready yet: {e}")
    if choice == "Demo form":
        st.info("Create it with: `python scripts/make_demo.py`")
    st.stop()

uploaded = st.file_uploader("Upload a filled form (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])
data, name, truth = None, None, None
if uploaded:
    data, name = uploaded.getvalue(), uploaded.name
elif choice == "Demo form" and DEMO_SAMPLES.exists():
    samples = sorted(p.name for p in DEMO_SAMPLES.glob("*.jpg"))
    if samples:
        pick = st.selectbox("...or try a demo sample", ["(none)"] + samples)
        if pick != "(none)":
            name, data = pick, (DEMO_SAMPLES / pick).read_bytes()
            tfile = (DEMO_SAMPLES / pick).with_suffix(".truth.json")
            if tfile.exists():
                truth = {t["name"]: t for t in json.loads(tfile.read_text())}

if data is None:
    st.stop()

with st.spinner("Analysing form..." + (" (the first run loads the OCR models)" if use_ocr else "")):
    page, out = run_vision(data, name, tpl_path, use_ocr, use_trocr)
base = None
if compare:
    with st.spinner("Running the baseline (full-page OCR)..."):
        base = run_baseline(data, name, tpl_path)

al = out.alignment
if al.ok:
    st.success(f"Aligned to template ({al.inliers} matching points)")
else:
    st.error("Could not align this form to the template; results may be wrong. "
             "Try a flatter, better-lit photo.")

tab_names = ["FormSense"] + (["Baseline (OCR + regex)", "Side by side"] if base else [])
tabs = st.tabs(tab_names)

with tabs[0]:
    col1, col2 = st.columns(2)
    col1.image(page[:, :, ::-1], caption="Uploaded", width="stretch")
    col2.image(annotate(out.aligned, out.fields)[:, :, ::-1],
               caption="Detected fields (green OK · red missing · orange unreadable · blue ticked)",
               width="stretch")
    st.dataframe(fields_table(out.fields), hide_index=True, width="stretch")
    st.download_button("Download field results (JSON)",
                       data="[" + ",".join(r.model_dump_json() for r in out.fields) + "]",
                       file_name=f"{Path(name).stem}_fields.json", mime="application/json")

if base:
    with tabs[1]:
        st.caption("What plain OCR sees: one block of text, no boxes, no ticks, no signatures. "
                   "Values are found by searching for each field's label.")
        col1, col2 = st.columns([2, 3])
        col1.code(base.text or "(no text found)", language=None)
        col2.dataframe(fields_table(base.fields).drop(columns=["Ink"]), hide_index=True,
                       width="stretch")

    with tabs[2]:
        kinds = {f.name: f.type for f in template.fields}
        b = {r.name: r for r in base.fields}
        rows, score = [], {"Baseline": 0, "FormSense": 0}
        for r in out.fields:
            row = {"Field": r.name}
            if truth:
                row["Correct answer"] = shown_truth(truth[r.name])
            for label, pred in (("Baseline", b[r.name]), ("FormSense", r)):
                cell = shown(pred)
                if truth:
                    ok = score_field(kinds[r.name], pred, truth[r.name])["correct"]
                    score[label] += ok
                    cell = ("✅ " if ok else "❌ ") + cell
                row[label] = cell
            rows.append(row)
        if truth:
            n = len(out.fields)
            c1, c2 = st.columns(2)
            c1.metric("Baseline: fields correct", f"{score['Baseline']}/{n}")
            c2.metric("FormSense: fields correct", f"{score['FormSense']}/{n}",
                      delta=score["FormSense"] - score["Baseline"])
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if kinds and any(k == FieldType.SIGNATURE for k in kinds.values()):
            st.caption("The baseline cannot see signatures or tick marks, so it assumes every form "
                       "is signed and guesses ticks from stray characters.")
