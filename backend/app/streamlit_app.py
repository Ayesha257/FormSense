"""FormSense demo UI. Run from the backend folder:

    streamlit run app/streamlit_app.py

Shows the vision step for now (alignment, field boxes, OCR). The rules,
LLM explanation and report plug in here as those modules are ready.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from formsense.ingest.loader import load_bytes  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.draw import annotate  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

TEMPLATES = {
    "Admission form": BACKEND / "data" / "templates" / "admission_form.json",
    "Demo form": BACKEND / "data" / "templates" / "demo" / "demo_template.json",
}
DEMO_SAMPLES = BACKEND / "data" / "templates" / "demo" / "samples"

st.set_page_config(page_title="FormSense", layout="wide")


@st.cache_resource
def get_template(path: str):
    return load_template(path)


@st.cache_resource
def get_ocr(use_trocr: bool):
    return FieldOCR(use_trocr=use_trocr)


st.title("FormSense: Form Auditor")

with st.sidebar:
    st.header("Settings")
    real_ready = (BACKEND / "data" / "templates" / "admission_form_blank.png").exists()
    choice = st.radio("Template", list(TEMPLATES), index=0 if real_ready else 1)
    use_ocr = st.checkbox("Run OCR", value=True)
    use_trocr = st.checkbox("Handwriting fallback (TrOCR)", value=True, disabled=not use_ocr)

try:
    template, blank = get_template(str(TEMPLATES[choice]))
except (FileNotFoundError, ValueError) as e:
    st.warning(f"{choice} is not ready yet: {e}")
    if choice == "Demo form":
        st.info("Create it with: `python scripts/make_demo.py`")
    st.stop()

uploaded = st.file_uploader("Upload a filled form (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])
data, name = None, None
if uploaded:
    data, name = uploaded.read(), uploaded.name
elif choice == "Demo form" and DEMO_SAMPLES.exists():
    samples = sorted(p.name for p in DEMO_SAMPLES.glob("*.jpg"))
    if samples:
        pick = st.selectbox("...or try a demo sample", ["(none)"] + samples)
        if pick != "(none)":
            name = pick
            data = (DEMO_SAMPLES / pick).read_bytes()

if data is None:
    st.stop()

page = load_bytes(data, name)[0]
with st.spinner("Analysing form..." + (" (first OCR run loads models)" if use_ocr else "")):
    ocr = get_ocr(use_trocr) if use_ocr else None
    out = analyze_page(page, template, blank, ocr=ocr)

al = out.alignment
if al.ok:
    st.success(f"Aligned to template ({al.inliers} matching points)")
else:
    st.error("Could not align this form to the template; results may be wrong. "
             "Try a flatter, better-lit photo.")

col1, col2 = st.columns(2)
col1.image(page[:, :, ::-1], caption="Uploaded", width="stretch")
col2.image(annotate(out.aligned, out.fields)[:, :, ::-1],
           caption="Detected fields (green OK · red missing · orange unreadable · blue ticked)",
           width="stretch")

rows = [{
    "Field": r.name,
    "Status": r.status.value,
    "Read as": r.raw_text or (r.value or ""),
    "OCR confidence": r.ocr_confidence,
    "Ink": r.ink_ratio,
} for r in out.fields]
st.subheader("Fields")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.download_button("Download field results (JSON)",
                   data="[" + ",".join(r.model_dump_json() for r in out.fields) + "]",
                   file_name=f"{Path(name).stem}_fields.json", mime="application/json")
