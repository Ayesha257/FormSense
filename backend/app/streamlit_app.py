"""FormSense demo UI. Run from the backend folder:

    python -m streamlit run app/streamlit_app.py

Upload a filled admission form (or pick a generated sample) and FormSense runs
the whole chain: vision (alignment, ink, OCR) -> rules (extract + validate)
-> the guideline behind every issue (RAG). Optionally the OCR + regex baseline
runs too, for a side-by-side comparison. The LLM explanation and the report
download plug in here when those modules are ready.
"""
import datetime as dt
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from formsense.baseline import BaselineAuditor  # noqa: E402
from formsense.ingest.loader import load_bytes  # noqa: E402
from formsense.pipeline import FormSense, issue_key  # noqa: E402
from formsense.rag import GuidelineRetriever  # noqa: E402
from formsense.rag.retriever import field_labels  # noqa: E402
from formsense.schema import FieldResult, FieldStatus  # noqa: E402
from formsense.scoring import score_field, score_issues, truth_from_label  # noqa: E402
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.draw import annotate  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

ADMISSION = BACKEND / "data" / "templates" / "admission_form.json"
DEMO = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
SAMPLES = {
    "Admission form": BACKEND / "data" / "synthetic" / "dev",
    "Demo form": BACKEND / "data" / "templates" / "demo" / "samples",
}
SAMPLE_HINT = {
    "Admission form": "python -m formsense.generator.generate --count 30 --split dev --seed 1 --out data/synthetic/dev",
    "Demo form": "python scripts/make_demo.py",
}
SEVERITY_ICON = {"error": "🔴", "warning": "🟠", "info": "🔵"}

st.set_page_config(page_title="FormSense", layout="wide")


# ---------------- cached models ----------------

@st.cache_resource
def get_template(path: str):
    return load_template(path)


@st.cache_resource
def get_ocr(use_trocr: bool):
    return FieldOCR(use_trocr=use_trocr)


@st.cache_resource
def get_retriever():
    return GuidelineRetriever().ensure()


@st.cache_resource
def get_baseline(path: str):
    template, _ = get_template(path)
    return BaselineAuditor(template, reader=get_ocr(False).paddle)


# ---------------- cached runs (per uploaded file) ----------------

@st.cache_data(show_spinner=False, max_entries=20)
def run_formsense(data: bytes, name: str, use_ocr: bool, use_trocr: bool, today: dt.date):
    fs = FormSense(ADMISSION, use_ocr=use_ocr, use_trocr=use_trocr,
                   ocr=get_ocr(use_trocr) if use_ocr else None, retriever=get_retriever())
    page = load_bytes(data, name)[0]
    return page, fs.run(page, today=today, form_id=Path(name).stem)


@st.cache_data(show_spinner=False, max_entries=20)
def run_vision_only(data: bytes, name: str, tpl_path: str, use_ocr: bool, use_trocr: bool):
    template, blank = get_template(tpl_path)
    page = load_bytes(data, name)[0]
    return page, analyze_page(page, template, blank, ocr=get_ocr(use_trocr) if use_ocr else None)


@st.cache_data(show_spinner=False, max_entries=20)
def run_baseline(data: bytes, name: str, tpl_path: str, today: dt.date | None):
    page = load_bytes(data, name)[0]
    fields = get_baseline(tpl_path).run(page).fields
    issues = None
    if today is not None:
        from formsense.rules.audit import audit
        issues = audit(fields, today=today).issues
    return fields, issues


# ---------------- display helpers ----------------

def shown(r: FieldResult) -> str:
    if r.checked is not None:
        return "☑ ticked" if r.checked else "☐ not ticked"
    if r.status == FieldStatus.MISSING:
        return "— missing"
    if r.status == FieldStatus.UNREADABLE:
        return f"? unreadable ({r.raw_text})" if r.raw_text else "? unreadable"
    text = r.raw_text or r.value or "present"
    return f"✗ {text}" if r.status == FieldStatus.INVALID else text


def shown_truth(t: dict) -> str:
    if "checked" in t:
        return "☑ ticked" if t["checked"] else "☐ not ticked"
    if t["status"] == "missing":
        return "— missing"
    return f"{t.get('text') or 'present'}" + ("" if t["status"] == "ok" else f" ({t['status']})")


def fields_table(results: list[FieldResult], labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Field": labels.get(r.name, r.name), "Status": r.status.value, "Read as": shown(r),
        "Value": r.value if r.checked is None else "", "OCR confidence": r.ocr_confidence,
        "Ink": r.ink_ratio,
    } for r in results])


def load_sample(folder: Path, pick: str):
    """(bytes, file name, truth fields, label dict) for a sample in either folder layout."""
    if (folder / "labels").is_dir():
        label = json.loads((folder / "labels" / f"{Path(pick).stem}.json").read_text(encoding="utf-8"))
        return (folder / "images" / pick).read_bytes(), pick, truth_from_label(label), label
    tfile = (folder / pick).with_suffix(".truth.json")
    truth = {t["name"]: t for t in json.loads(tfile.read_text())} if tfile.exists() else None
    return (folder / pick).read_bytes(), pick, truth, None


def list_samples(folder: Path) -> list[str]:
    if (folder / "images").is_dir():
        return sorted(p.name for p in (folder / "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []


# ---------------- page ----------------

st.title("FormSense: Form Auditor")

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Template", ["Admission form", "Demo form"])
    use_ocr = st.checkbox("Run OCR", value=True)
    use_trocr = st.checkbox("Handwriting fallback (TrOCR)", value=True, disabled=not use_ocr)
    compare = st.checkbox("Compare with baseline (OCR + regex)", value=False,
                          help="Runs plain full-page OCR too. Slower: about 20-40 s per form on a laptop CPU.")
    today = st.date_input("Check dates against", value=dt.date.today(),
                          help="Used by date rules (age, future dates). Samples use their own reference date.")

tpl_path = str(ADMISSION if choice == "Admission form" else DEMO)
try:
    template, blank = get_template(tpl_path)
except (FileNotFoundError, ValueError) as e:
    st.warning(f"{choice} is not ready yet: {e}")
    if choice == "Demo form":
        st.info("Create it with: `python scripts/make_demo.py`")
    st.stop()
labels = field_labels(tpl_path) if choice == "Admission form" else {f.name: f.label for f in template.fields}
kinds = {f.name: f.type for f in template.fields}

uploaded = st.file_uploader("Upload a filled form (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])
data = name = truth = label = None
if uploaded:
    data, name = uploaded.getvalue(), uploaded.name
else:
    samples = list_samples(SAMPLES[choice])
    if samples:
        pick = st.selectbox("...or try a sample", ["(none)"] + samples)
        if pick != "(none)":
            data, name, truth, label = load_sample(SAMPLES[choice], pick)
    else:
        st.caption(f"No sample forms yet. Create some with: `{SAMPLE_HINT[choice]}`")
if data is None:
    st.stop()

ref_date = dt.date.fromisoformat(label["reference_date"]) if label else today
audit_on = choice == "Admission form"
with st.spinner("Checking the form..." + (" (the first run loads the OCR models)" if use_ocr else "")):
    if audit_on:
        page, result = run_formsense(data, name, use_ocr, use_trocr, ref_date)
        vision, fields, issues = result.vision, result.fields, result.issues
    else:
        page, vision = run_vision_only(data, name, tpl_path, use_ocr, use_trocr)
        result, fields, issues = None, vision.fields, []
base_fields = base_issues = None
if compare:
    with st.spinner("Running the baseline (full-page OCR)..."):
        base_fields, base_issues = run_baseline(data, name, tpl_path, ref_date if audit_on else None)

al = vision.alignment
if not al.ok:
    st.error("Could not align this form to the template; results may be wrong. "
             "Try a flatter, better-lit photo.")
elif audit_on:
    n_err = sum(i.severity.value == "error" for i in issues)
    n_warn = len(issues) - n_err
    if not issues:
        st.success("No problems found: the form looks complete and consistent.")
    else:
        st.warning(f"{n_err} error(s) and {n_warn} warning(s) found.")

tab_names = (["Audit"] if audit_on else []) + ["Fields"]
if base_fields is not None:
    tab_names += ["Baseline (OCR + regex)", "Side by side"]
tab_names += ["Guidelines"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ----- Audit: the issues, each with the guideline behind it -----
if audit_on:
    with tabs["Audit"]:
        col1, col2 = st.columns([5, 6])
        col1.image(annotate(vision.aligned, fields)[:, :, ::-1], width="stretch",
                   caption="green OK · red missing · orange unreadable · purple invalid · blue ticked")
        with col2:
            truth_keys = None
            if label:
                found = score_issues(issues, label["issues"])
                truth_keys = {f"{i['id']}:{','.join(i['fields'])}" for i in label["issues"]}
                c1, c2, c3 = st.columns(3)
                c1.metric("True issues", len(label["issues"]))
                c2.metric("Found", found["tp"])
                c3.metric("False alarms", found["fp"])
            if not issues:
                st.info("No issues.")
            for i in issues:
                mark = ""
                if truth_keys is not None:
                    mark = "✅ " if issue_key(i) in truth_keys else "➕ (not a real issue) "
                fields_txt = ", ".join(dict.fromkeys(labels.get(f, f) for f in i.fields))
                with st.expander(f"{mark}{SEVERITY_ICON.get(i.severity.value, '')} {fields_txt}: {i.message}"):
                    st.caption(f"rule `{i.id}` · {i.type.value} · {i.severity.value}")
                    cite = result.citations.get(issue_key(i))
                    if cite:
                        st.markdown(f"**Guideline** · *{cite.source}*")
                        st.write(cite.text)
                    else:
                        st.write("No matching guideline found.")
            if label:
                missed = sorted(truth_keys - {issue_key(i) for i in issues})
                for k in missed:
                    rule, flds = k.split(":", 1)
                    st.error(f"❌ Missed: `{rule}` on {', '.join(labels.get(f, f) for f in flds.split(','))}")
        st.download_button("Download audit (JSON)", data=result.report().model_dump_json(indent=2),
                           file_name=f"{Path(name).stem}_audit.json", mime="application/json")

# ----- Fields -----
with tabs["Fields"]:
    if not audit_on:
        col1, col2 = st.columns(2)
        col1.image(page[:, :, ::-1], caption="Uploaded", width="stretch")
        col2.image(annotate(vision.aligned, fields)[:, :, ::-1], width="stretch",
                   caption="green OK · red missing · orange unreadable · blue ticked")
    df = fields_table(fields, labels)
    if truth:
        df.insert(1, "Correct answer", [shown_truth(truth[r.name]) for r in fields])
        df.insert(2, "✓", ["✅" if score_field(kinds[r.name], r, truth[r.name])["correct"] else "❌"
                           for r in vision.fields])
    st.dataframe(df, hide_index=True, width="stretch")
    if audit_on and result.notes:
        with st.expander("Extraction notes"):
            for f, notes in result.notes.items():
                st.write(f"**{labels.get(f, f)}**: " + "; ".join(notes))

# ----- Baseline and comparison -----
if base_fields is not None:
    with tabs["Baseline (OCR + regex)"]:
        st.caption("What plain OCR sees: values found by searching the page text for each field's label. "
                   "No boxes, no ticks, no signatures.")
        st.dataframe(fields_table(base_fields, labels).drop(columns=["Ink", "Value"]),
                     hide_index=True, width="stretch")

    with tabs["Side by side"]:
        b = {r.name: r for r in base_fields}
        v = {r.name: r for r in vision.fields}
        score = {"Baseline": 0, "FormSense": 0}
        rows = []
        for f in template.fields:
            row = {"Field": labels.get(f.name, f.name)}
            if truth:
                row["Correct answer"] = shown_truth(truth[f.name])
            for who, pred in (("Baseline", b[f.name]), ("FormSense", v[f.name])):
                cell = shown(pred)
                if truth:
                    ok = score_field(f.type, pred, truth[f.name])["correct"]
                    score[who] += ok
                    cell = ("✅ " if ok else "❌ ") + cell
                row[who] = cell
            rows.append(row)
        if truth:
            n = len(template.fields)
            c1, c2 = st.columns(2)
            c1.metric("Baseline: fields correct", f"{score['Baseline']}/{n}")
            c2.metric("FormSense: fields correct", f"{score['FormSense']}/{n}",
                      delta=score["FormSense"] - score["Baseline"])
            if label and base_issues is not None:
                bi = score_issues(base_issues, label["issues"])
                fi = score_issues(issues, label["issues"])
                c1.metric("Baseline: true issues found", f"{bi['tp']}/{len(label['issues'])}",
                          help=f"plus {bi['fp']} false alarms")
                c2.metric("FormSense: true issues found", f"{fi['tp']}/{len(label['issues'])}",
                          delta=fi["tp"] - bi["tp"], help=f"plus {fi['fp']} false alarms")
                st.caption(f"False alarms: baseline {bi['fp']}, FormSense {fi['fp']}.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ----- Guidelines search -----
with tabs["Guidelines"]:
    try:
        retriever = get_retriever()
    except FileNotFoundError as e:
        st.warning(f"No guidelines found: {e}")
        retriever = None
    if retriever:
        st.caption("Search over data/guidelines/ (LangChain + FAISS, hybrid semantic + keyword search). "
                   "The same retrieval attaches a guideline to every issue in the Audit tab.")
        q = st.text_input("Ask the guidelines", placeholder="e.g. Can my mother sign the form for me?")
        if q:
            for p in retriever.search(q, k=2):
                with st.container(border=True):
                    st.markdown(f"**{p.section}** · score {p.score:.2f}")
                    st.write(p.text)
