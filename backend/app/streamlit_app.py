"""FormSense demo UI. Run from the backend folder:

    python -m streamlit run app/streamlit_app.py

Upload a filled admission form (or pick a generated sample). FormSense runs the whole
chain: vision (alignment, ink, OCR) -> rules (extract + validate) -> the guideline
behind every issue (RAG). Optionally the OCR + regex baseline and a vision-language
model (Groq) read the same form for a side-by-side comparison. When the language side
provides them, an LLM explanation and a PDF report appear automatically.

Tabs: Overview (verdict, explanation, downloads) · Issues (with guidelines and evidence)
· Checklist · Fields (with a per-field inspector) · Compare · Guidelines.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from formsense.baseline import BaselineAuditor  # noqa: E402
from formsense.ingest.loader import load_bytes  # noqa: E402
from formsense.pipeline import FormSense, issue_key  # noqa: E402
from formsense.rag import GuidelineRetriever  # noqa: E402
from formsense.rag.retriever import field_labels  # noqa: E402
from formsense.schema import FieldResult, FieldStatus  # noqa: E402
from formsense.scoring import score_field, score_issues, truth_from_label  # noqa: E402
from formsense.explain import (LLMModelUnavailable, build_full_report, explain_report,  # noqa: E402
                               has_llm_key, is_offline, pdf_bytes)
from formsense.summary import (STATUS_ICON, checklist, checklist_csv, crop_field, field_counts,  # noqa: E402
                               highlight, issue_counts, plain_summary, png_bytes, sort_issues)
from formsense.vision import analyze_page, load_template  # noqa: E402
from formsense.vision.draw import annotate  # noqa: E402
from formsense.vision.ocr import FieldOCR  # noqa: E402

ADMISSION = BACKEND / "data" / "templates" / "admission_form.json"
DEMO = BACKEND / "data" / "templates" / "demo" / "demo_template.json"
SAMPLES = {
    "Admission form": Path(os.getenv("FORMSENSE_SAMPLE_DIR", BACKEND / "data" / "synthetic" / "dev")),
    "Demo form": BACKEND / "data" / "templates" / "demo" / "samples",
}
SAMPLE_HINT = {
    "Admission form": "python -m formsense.generator.generate --count 30 --split dev --seed 1 --out data/synthetic/dev",
    "Demo form": "python scripts/make_demo.py",
}
SEVERITY_ICON = {"error": "🔴", "warning": "🟠", "info": "🔵"}
LEGEND = "green OK · red missing · orange unreadable · purple invalid · blue ticked"

st.set_page_config(page_title="FormSense", page_icon="📝", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border-radius: 10px; padding: .6rem .9rem;}
  .verdict {border-radius: 10px; padding: .8rem 1.1rem; font-size: 1.05rem; margin-bottom: .6rem;}
  .verdict.good {background: rgba(34,197,94,.14);} .verdict.warn {background: rgba(245,158,11,.16);}
  .verdict.bad {background: rgba(239,68,68,.14);}
</style>""", unsafe_allow_html=True)


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


@st.cache_resource
def get_vlm(path: str, model: str):
    from formsense.vlm import VLMAuditor
    template, _ = get_template(path)
    return VLMAuditor(template, model=model or None)


def groq_key_present() -> bool:
    if os.getenv("GROQ_API_KEY"):
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except ImportError:
        return False
    return bool(os.getenv("GROQ_API_KEY"))


# ---------------- cached runs (per uploaded file) ----------------

def get_page(data: bytes, name: str, page_idx: int):
    pages = load_bytes(data, name)
    return pages[min(page_idx, len(pages) - 1)]


@st.cache_data(show_spinner=False)
def count_pages(data: bytes, name: str) -> int:
    return len(load_bytes(data, name))


@st.cache_data(show_spinner=False, max_entries=20)
def run_formsense(data: bytes, name: str, page_idx: int, use_ocr: bool, use_trocr: bool, today: dt.date):
    fs = FormSense(ADMISSION, use_ocr=use_ocr, use_trocr=use_trocr,
                   ocr=get_ocr(use_trocr) if use_ocr else None, retriever=get_retriever())
    page = get_page(data, name, page_idx)
    return page, fs.run(page, today=today, form_id=Path(name).stem)


@st.cache_data(show_spinner=False, max_entries=20)
def run_vision_only(data: bytes, name: str, page_idx: int, tpl_path: str, use_ocr: bool, use_trocr: bool):
    template, blank = get_template(tpl_path)
    page = get_page(data, name, page_idx)
    return page, analyze_page(page, template, blank, ocr=get_ocr(use_trocr) if use_ocr else None)


def with_issues(fields, today: dt.date | None):
    if today is None:
        return fields, None
    from formsense.rules.audit import audit
    return fields, audit(fields, today=today).issues


@st.cache_data(show_spinner=False, max_entries=20)
def run_baseline(data: bytes, name: str, page_idx: int, tpl_path: str, today: dt.date | None):
    page = get_page(data, name, page_idx)
    return with_issues(get_baseline(tpl_path).run(page).fields, today)


@st.cache_data(show_spinner=False, max_entries=20)
def run_vlm(data: bytes, name: str, page_idx: int, tpl_path: str, model: str, today: dt.date | None):
    page = get_page(data, name, page_idx)
    return with_issues(get_vlm(tpl_path, model).run(page).fields, today)


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


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def evidence_boxes(issue, by_name):
    boxes = [tuple(b) for b in issue.evidence] or [by_name[f].bbox for f in issue.fields if f in by_name]
    return [b for b in boxes if b[2] > 0 and b[3] > 0]


# ---------------- sidebar ----------------

st.title("📝 FormSense: Form Auditor")
st.caption("Upload a filled admission form. FormSense checks it is complete, consistent and valid, "
           "then shows what to fix and which guideline says so.")

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Template", ["Admission form", "Demo form"])
    use_ocr = st.checkbox("Run OCR", value=True)
    use_trocr = st.checkbox("Handwriting fallback (TrOCR)", value=True, disabled=not use_ocr)
    compare = st.checkbox("Compare with baseline (OCR + regex)", value=False,
                          help="Runs plain full-page OCR too. Slower: about 20-40 s per form on a laptop CPU.")
    has_key = groq_key_present()
    compare_vlm = st.checkbox("Compare with a vision-language model (Groq)", value=False, disabled=not has_key,
                              help="Sends the form image to a Groq vision model (Qwen 3.8 27B by default) and compares its reading. "
                                   "The image leaves your machine: use synthetic forms.")
    vlm_model = st.text_input("VLM model id (optional)", value="", disabled=not has_key or not compare_vlm,
                              placeholder="default: GROQ_VLM_MODEL or Qwen 3.8 27B",
                              help="Groq retires models from time to time. Paste a current vision model id "
                                   "here, or set GROQ_VLM_MODEL in backend/.env. "
                                   "Run `python scripts/groq_models.py` to list what your key can use.")
    if not has_key:
        st.caption("Add GROQ_API_KEY to backend/.env to enable the VLM comparison.")
    use_rag_explain = st.checkbox("Use the guidelines in the explanation (RAG)", value=True,
                                  help="The retrieved guideline passages are given to the LLM. "
                                       "Off = the LLM sees only the issues.")
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

# ---------------- input ----------------

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

try:
    n_pages = count_pages(data, name)
except Exception as e:  # corrupt PDF / unreadable image
    st.error(f"This file could not be opened: {e}")
    st.stop()
page_idx = 0
if n_pages > 1:
    page_idx = st.sidebar.number_input("Page of the PDF", 1, n_pages, 1, help="The form is checked one page at a time.") - 1

ref_date = dt.date.fromisoformat(label["reference_date"]) if label else today
audit_on = choice == "Admission form"
try:
    with st.spinner("Checking the form..." + (" (the first run loads the OCR models)" if use_ocr else "")):
        if audit_on:
            page, result = run_formsense(data, name, page_idx, use_ocr, use_trocr, ref_date)
            vision, fields, issues = result.vision, result.fields, result.issues
        else:
            page, vision = run_vision_only(data, name, page_idx, tpl_path, use_ocr, use_trocr)
            result, fields, issues = None, vision.fields, []
except Exception as e:
    st.error(f"Something went wrong while checking this form: {e}")
    st.stop()

base_fields = base_issues = vlm_fields = vlm_issues = None
vlm_error = None
if compare:
    with st.spinner("Running the baseline (full-page OCR)..."):
        base_fields, base_issues = run_baseline(data, name, page_idx, tpl_path, ref_date if audit_on else None)
if compare_vlm:
    with st.spinner("Asking the vision-language model..."):
        try:
            vlm_fields, vlm_issues = run_vlm(data, name, page_idx, tpl_path, vlm_model.strip(), ref_date if audit_on else None)
        except Exception as e:
            vlm_error = str(e)

al = vision.alignment
if not al.ok:
    st.error("Could not align this form to the template; results may be wrong. "
             "Try a flatter, better-lit photo.")
if not use_ocr:
    st.info("OCR is off: only ticks, signatures and empty boxes are detected. Written values are not read.")

annotated = annotate(vision.aligned, fields)
counts = issue_counts(issues)
fc = field_counts(fields, template)

# ---------------- verdict + metrics ----------------

if audit_on and al.ok:
    if not issues:
        st.markdown('<div class="verdict good">✅ <b>No problems found.</b> The form looks complete and consistent.</div>',
                    unsafe_allow_html=True)
    elif counts["error"]:
        st.markdown(f'<div class="verdict bad">⛔ <b>Needs fixing.</b> {counts["error"]} error(s) and '
                    f'{counts["warning"]} warning(s) found.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict warn">⚠️ <b>Almost there.</b> {counts["warning"]} warning(s) to check.</div>',
                    unsafe_allow_html=True)
m = st.columns(5)
m[0].metric("Items OK", fc["ok"])
m[1].metric("Missing", fc["missing"])
m[2].metric("Unreadable", fc["unreadable"])
m[3].metric("Invalid", fc["invalid"])
if audit_on:
    m[4].metric("Issues", len(issues), help=f"{counts['error']} errors, {counts['warning']} warnings, {counts['info']} notes")

tab_names = (["Overview", "Issues"] if audit_on else []) + ["Checklist", "Fields"]
if compare or compare_vlm:
    tab_names += ["Compare"]
tab_names += ["Guidelines"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ----- Overview -----
if audit_on:
    with tabs["Overview"]:
        col1, col2 = st.columns([5, 6], gap="large")
        with col1:
            ordered = sort_issues(issues)
            options = {f"{n}. {', '.join(dict.fromkeys(labels.get(f, f) for f in i.fields))}: {i.message[:70]}": i
                       for n, i in enumerate(ordered, 1)}
            focus = st.selectbox("Highlight an issue on the form", ["(all fields)"] + list(options), key="focus")
            img = to_bgr(annotated)
            if focus in options:
                img = highlight(img, evidence_boxes(options[focus], {r.name: r for r in fields}))
            st.image(img[:, :, ::-1], width="stretch", caption=LEGEND)
        with col2:
            skey = f"explanation::{name}::{page_idx}::{ref_date}::{use_rag_explain}"
            explanation = st.session_state.get(skey)
            st.subheader("What to fix")
            if explanation is not None:
                st.write(explanation.summary)
                for e in explanation.issues:
                    st.markdown(f"- {e.explanation}")
                if explanation.overall_recommendation:
                    st.info(explanation.overall_recommendation)
                st.caption(("Template text (no GROQ_API_KEY set), not written by an LLM."
                            if st.session_state.get(skey + "::offline") else
                            "Written by the LLM from the audit"
                            + (" and the guidelines." if use_rag_explain else " alone (no guidelines).")))
            else:
                st.markdown(plain_summary(fields, issues, template, labels))
            if issues:
                if st.button("Write a plain-English explanation" + ("" if has_llm_key() else " (offline)"),
                             key="explain"):
                    with st.spinner("Writing the explanation..."):
                        try:
                            st.session_state[skey + "::offline"] = is_offline()
                            st.session_state[skey] = explain_report(result.report(), ADMISSION,
                                                                    use_rag=use_rag_explain)
                            st.rerun()
                        except LLMModelUnavailable as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"The explanation could not be written: {e}")
            if not has_llm_key():
                st.caption("No GROQ_API_KEY in backend/.env: explanations use a plain template.")

            st.subheader("Download")
            report = build_full_report(result, explanation)
            rows = checklist(fields, template, labels)
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("JSON", data=report.model_dump_json(indent=2),
                               file_name=f"{Path(name).stem}_audit.json", mime="application/json", width="stretch")
            d2.download_button("Image", data=png_bytes(annotated), file_name=f"{Path(name).stem}_annotated.png",
                               mime="image/png", width="stretch")
            d3.download_button("Checklist", data=checklist_csv(rows), file_name=f"{Path(name).stem}_checklist.csv",
                               mime="text/csv", width="stretch")
            try:
                d4.download_button("PDF", data=pdf_bytes(report, ADMISSION), file_name=f"{Path(name).stem}_audit.pdf",
                                   mime="application/pdf", width="stretch")
            except Exception as e:
                d4.button("PDF", disabled=True, width="stretch", help=f"The PDF could not be built: {e}")

# ----- Issues -----
if audit_on:
    with tabs["Issues"]:
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
        by_name = {r.name: r for r in fields}
        aligned_bgr = to_bgr(vision.aligned)
        for n, i in enumerate(sort_issues(issues)):
            mark = ""
            if truth_keys is not None:
                mark = "✅ " if issue_key(i) in truth_keys else "➕ (not a real issue) "
            fields_txt = ", ".join(dict.fromkeys(labels.get(f, f) for f in i.fields))
            with st.expander(f"{mark}{SEVERITY_ICON.get(i.severity.value, '')} {fields_txt}: {i.message}"):
                st.caption(f"rule `{i.id}` · {i.type.value} · {i.severity.value}")
                boxes = evidence_boxes(i, by_name)[:3]
                if boxes:
                    cols = st.columns(len(boxes))
                    for c, b in zip(cols, boxes):
                        c.image(crop_field(aligned_bgr, b)[:, :, ::-1], width="stretch")
                cite = result.citations.get(issue_key(i))
                if cite:
                    st.markdown(f"**Guideline** · *{cite.source}*")
                    st.write(cite.text)
                else:
                    st.write("No matching guideline found.")
        if label:
            for k in sorted(truth_keys - {issue_key(i) for i in issues}):
                rule, flds = k.split(":", 1)
                st.error(f"❌ Missed: `{rule}` on {', '.join(labels.get(f, f) for f in flds.split(','))}")

# ----- Checklist -----
with tabs["Checklist"]:
    rows = checklist(fields, template, labels)
    st.caption("One line per item on the form. A tick-box group (gender, program) is ok when exactly one box is ticked.")
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            st.markdown(f"**{section}**")
        st.markdown(f"{STATUS_ICON.get(r['status'], '')} **{r['item']}** · {r['detail']}")

# ----- Fields -----
with tabs["Fields"]:
    if not audit_on:
        col1, col2 = st.columns(2)
        col1.image(page[:, :, ::-1], caption="Uploaded", width="stretch")
        col2.image(annotated[:, :, ::-1], width="stretch", caption=LEGEND)
    df = fields_table(fields, labels)
    if truth:
        df.insert(1, "Correct answer", [shown_truth(truth[r.name]) for r in fields])
        df.insert(2, "✓", ["✅" if score_field(kinds[r.name], r, truth[r.name])["correct"] else "❌"
                           for r in vision.fields])
    st.dataframe(df, hide_index=True, width="stretch")
    st.markdown("**Field inspector**")
    pick_field = st.selectbox("Look closely at a field", [r.name for r in fields],
                              format_func=lambda n: labels.get(n, n), key="inspect")
    r = next(x for x in fields if x.name == pick_field)
    ic1, ic2 = st.columns([3, 2])
    ic1.image(crop_field(to_bgr(vision.aligned), r.bbox, pad=20)[:, :, ::-1], width="stretch",
              caption="what FormSense saw in this box")
    ic2.markdown(f"**Status:** {STATUS_ICON.get(r.status.value, '')} {r.status.value}  \n"
                 f"**Read as:** {shown(r)}  \n"
                 f"**OCR confidence:** {r.ocr_confidence if r.ocr_confidence is not None else 'n/a'}  \n"
                 f"**Ink in box:** {r.ink_ratio:.1%}")
    if audit_on and result.notes:
        with st.expander("Extraction notes"):
            for f, notes in result.notes.items():
                st.write(f"**{labels.get(f, f)}**: " + "; ".join(notes))

# ----- Compare -----
if "Compare" in tabs:
    with tabs["Compare"]:
        systems = {}
        if base_fields is not None:
            systems["Baseline"] = (base_fields, base_issues)
        if vlm_fields is not None:
            systems["VLM (Groq)"] = (vlm_fields, vlm_issues)
        systems["FormSense"] = (fields, issues)
        if vlm_error:
            st.error(f"The VLM could not read this form: {vlm_error}")
        st.caption("Baseline: plain full-page OCR + regex, no boxes, ticks or signatures. VLM: the whole image "
                   "sent to a vision-language model; it gives no confidence score and no pixel evidence.")
        if len(systems) > 1:
            lookups = {who: {r.name: r for r in flds} for who, (flds, _) in systems.items()}
            score = {who: 0 for who in systems}
            rows = []
            for f in template.fields:
                row = {"Field": labels.get(f.name, f.name)}
                if truth:
                    row["Correct answer"] = shown_truth(truth[f.name])
                for who in systems:
                    cell = shown(lookups[who][f.name])
                    if truth:
                        ok = score_field(f.type, lookups[who][f.name], truth[f.name])["correct"]
                        score[who] += ok
                        cell = ("✅ " if ok else "❌ ") + cell
                    row[who] = cell
                rows.append(row)
            if truth:
                n = len(template.fields)
                cols = st.columns(len(systems))
                for c, (who, (_, iss)) in zip(cols, systems.items()):
                    c.metric(f"{who}: fields correct", f"{score[who]}/{n}")
                    if label and iss is not None:
                        s = score_issues(iss, label["issues"])
                        c.metric(f"{who}: true issues found", f"{s['tp']}/{len(label['issues'])}",
                                 help=f"plus {s['fp']} false alarms")
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
                   "The same retrieval attaches a guideline to every issue in the Issues tab.")
        q = st.text_input("Ask the guidelines", placeholder="e.g. Can my mother sign the form for me?")
        if q:
            for p in retriever.search(q, k=2):
                with st.container(border=True):
                    st.markdown(f"**{p.section}** · score {p.score:.2f}")
                    st.write(p.text)
