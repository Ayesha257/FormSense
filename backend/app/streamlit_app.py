"""FormSense demo UI. Run: streamlit run app/streamlit_app.py

Owner: Ayesha. For now it only uploads and shows the cleaned page;
the pipeline gets plugged in as modules are ready.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from formsense.ingest.loader import load_bytes, preprocess

st.set_page_config(page_title="FormSense", layout="wide")
st.title("FormSense: Form Auditor")
st.caption("Upload a filled admission form (PDF or image).")

uploaded = st.file_uploader("Form", type=["pdf", "png", "jpg", "jpeg"])
if uploaded:
    pages = load_bytes(uploaded.read(), uploaded.name)
    st.write(f"{len(pages)} page(s) loaded")
    for i, page in enumerate(pages, 1):
        col1, col2 = st.columns(2)
        col1.image(page[:, :, ::-1], caption=f"Page {i}: original")
        col2.image(preprocess(page), caption=f"Page {i}: cleaned", clamp=True)
    st.info("Field detection, checks and the audit report will appear here.")
