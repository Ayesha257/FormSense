"""Rate LLM explanations 1-5 (faithfulness, clarity, usefulness). One rater per session.

    python -m streamlit run app/rate_explanations.py

Build the items first with scripts/make_rating_set.py. Items are blinded: you
never see whether an explanation came from the LLM alone or from LLM + RAG.
Scores are saved to eval/rating/ratings_<your name>.json after every item, so
you can stop and continue any time. Afterwards, scripts/rating_report.py
combines two raters' files into agreement and mean scores.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import streamlit as st  # noqa: E402

from formsense import rating as R  # noqa: E402
from formsense.rag.retriever import GUIDELINES_DIR, guideline_files  # noqa: E402

st.set_page_config(page_title="FormSense: rate explanations", layout="wide")

SEVERITY_ICON = {"error": "🔴", "warning": "🟠", "info": "🔵"}
LABELS = {1: "1 · poor", 2: "2", 3: "3 · okay", 4: "4", 5: "5 · excellent"}

st.title("Rate explanations")

with st.sidebar:
    st.header("Rater")
    rater = st.text_input("Your name", placeholder="e.g. Ayesha")
    folder = Path(st.text_input("Rating folder", value=str(R.RATING_DIR)))
    with st.expander("Rating guide", expanded=False):
        for d, text in R.RUBRIC.items():
            st.markdown(f"**{d.title()}**  \n{text}")

set_file = folder / "rating_set.json"
if not set_file.exists():
    st.warning(f"No rating set at `{set_file}`.")
    st.code("python scripts/make_rating_set.py --dir data/synthetic/test --limit 15")
    st.stop()
try:
    items = R.load_items(set_file)
except (ValueError, KeyError, OSError) as e:
    st.error(f"The rating set could not be read: {e}")
    st.stop()
if not rater.strip():
    st.info("Enter your name in the sidebar to start. Your scores are saved under that name.")
    st.stop()

path = R.ratings_path(rater, folder)
doc = R.load_ratings(path)
done, total = R.progress(items, doc)
with st.sidebar:
    st.progress(done / total if total else 0.0, text=f"{done} of {total} rated")
    if path.exists():
        st.download_button("Download my ratings", data=path.read_bytes(), file_name=path.name,
                           mime="application/json")

if st.session_state.get("rater") != rater or st.session_state.get("folder") != str(folder):
    first_open = next((n for n, i in enumerate(items) if not R.is_complete(doc["ratings"].get(i["id"], {}))), 0)
    st.session_state.update(rater=rater, folder=str(folder), pos=first_open)
pos = min(max(st.session_state.get("pos", 0), 0), len(items) - 1)

if '"offline_test_set": true' in set_file.read_text(encoding="utf-8"):
    st.warning("TEST SET (offline, templated text). Do not use these ratings for results.")
if done == total:
    st.success("All items are rated. Send your ratings file to the other rater, then run "
               "`python scripts/rating_report.py <file_a> <file_b>`.")

# ---- jump bar ----
jump = st.columns([1, 1, 1, 3])
if jump[0].button("◀ Previous", disabled=pos == 0, width="stretch"):
    st.session_state.pos = pos - 1
    st.rerun()
if jump[1].button("Next ▶", disabled=pos >= len(items) - 1, width="stretch"):
    st.session_state.pos = pos + 1
    st.rerun()
todo = [n for n, i in enumerate(items) if not R.is_complete(doc["ratings"].get(i["id"], {}))]
if jump[2].button("Next unrated", disabled=not todo, width="stretch"):
    st.session_state.pos = next((n for n in todo if n > pos), todo[0])
    st.rerun()

item = items[pos]
existing = doc["ratings"].get(item["id"], {})
st.subheader(f"Item {pos + 1} of {len(items)}" + ("  ✅ rated" if R.is_complete(existing) else ""))

left, right = st.columns([6, 5], gap="large")
with left:
    image = folder / item["image"] if item.get("image") else None
    if image is not None and image.exists():
        st.image(str(image), width="stretch", caption="the form, with FormSense's field boxes")
    st.markdown("**What the checker found**")
    for i in item["issues"]:
        st.markdown(f"{SEVERITY_ICON.get(i['severity'], '')} {i['message']}")
    if item.get("facts"):
        with st.expander("What was read from the form"):
            st.write("\n".join(f"- {f}" for f in item["facts"]))
    with st.expander("University guidelines (for checking faithfulness)"):
        for f in guideline_files(GUIDELINES_DIR):
            st.markdown(f.read_text(encoding="utf-8"))

with right:
    st.markdown("**Explanation to rate**")
    with st.container(border=True):
        st.write(item["explanation"])
    with st.form(f"rate_{item['id']}"):
        scores = {}
        for d in R.DIMENSIONS:
            scores[d] = st.radio(d.title(), R.SCALE, index=R.SCALE.index(existing[d]) if existing.get(d) in R.SCALE else None,
                                 format_func=LABELS.get, horizontal=True, key=f"{d}_{item['id']}",
                                 help=R.RUBRIC[d])
        comment = st.text_area("Comment (optional)", value=existing.get("comment", ""), key=f"c_{item['id']}",
                               height=80)
        saved = st.form_submit_button("Save and next", type="primary", width="stretch")
    if saved:
        if any(v is None for v in scores.values()):
            st.error("Score all three dimensions before saving.")
        else:
            R.save_rating(path, rater.strip(), item["id"], scores, comment)
            fresh = R.load_ratings(path)
            left_todo = [n for n, i in enumerate(items) if not R.is_complete(fresh["ratings"].get(i["id"], {}))]
            st.session_state.pos = next((n for n in left_todo if n > pos),
                                        left_todo[0] if left_todo else min(pos + 1, len(items) - 1))
            st.rerun()
