# Week 7: UI polish, VLM comparison, explanation ratings

Run every command from `backend/`.

## 1. The app

```bash
python -m streamlit run app/streamlit_app.py
```

| Tab | What it shows |
| --- | --- |
| Overview | Verdict banner, form with coloured boxes (pick an issue to highlight where it is), what to fix, downloads |
| Issues | Every issue, errors first, with a crop of the evidence and the guideline behind it |
| Checklist | One line per form item. A tick-box group is ok only if exactly one box is ticked |
| Fields | All readings, a correct-answer column on generated samples, and a per-field inspector |
| Compare | Baseline and/or VLM next to FormSense (sidebar checkboxes) |
| Guidelines | Free-text search over the guidelines (RAG) |

Downloads: JSON audit, annotated image (PNG), checklist (CSV). **PDF** and the
**LLM explanation** button appear on their own when these exist:

| Hook | Where the app looks | Contract |
| --- | --- | --- |
| LLM explainer | `formsense.llm.explain` (also `formsense.llm.explainer.explain`) | `explain(report: AuditReport) -> str \| AuditReport`. `report.citations` filled = LLM + RAG |
| PDF report | `formsense.report.build_pdf` | `build_pdf(report: AuditReport, annotated_bgr: np.ndarray \| None) -> bytes` |

Easha: expose those two functions (or tell Ayesha the real names, the lookup is one
line each in `formsense/summary.py`) and the buttons light up with no UI change.
Multi-page PDFs get a page selector. Errors (bad file, alignment failure, VLM or
LLM failure) show a message instead of a stack trace.

## 2. VLM comparison (stretch goal)

Needs `GROQ_API_KEY` in `backend/.env`. The default vision model is `qwen/qwen3.8-27b`
(Groq's current vision model, run in instruct mode). **Llama 4 Scout, the model in the original
roadmap, was shut down by Groq on 2026-07-17**, and Groq retires models regularly, so the model id is
a setting, not code: put `GROQ_VLM_MODEL=<model id>` in `.env`, or use `--model`, or the "VLM model id"
box in the app sidebar. Check what your key can use, and test it, before a long run:

```bash
python scripts/groq_models.py            # models your key can use
python scripts/groq_models.py --check    # sends a tiny test image to the configured vision model
```

```bash
# 1. no key needed: checks the script end to end with a SIMULATED VLM (numbers are not results)
python scripts/compare_vlm.py --dir data/synthetic/test --limit 5 --dry-run

# 2. the real comparison on the held-out forms (answers are cached under outputs/vlm_cache,
#    so a re-run costs nothing; --delay keeps you under the free-tier requests-per-minute cap)
python scripts/compare_vlm.py --dir data/synthetic/test --limit 20

# only the VLM (fast, no OCR models loaded)
python scripts/compare_vlm.py --dir data/synthetic/test --systems vlm
```

Output: `outputs/comparison_vlm/summary.md` (paste into the results slide), `summary.json`,
`fields.csv`, `issues.csv`. All three systems go through the same rules and scoring.
Metrics use only forms every system answered; failures are listed under the table.

How to talk about it: the VLM needs no template or training and reads clean
text well, but gives no confidence score and no pixel evidence for ticks and
signatures, can answer differently between runs, and sends every form to a third
party. FormSense gives a per-field confidence, separates missing from unreadable by ink
density, and keeps the image on the machine. Report whatever the table shows; do not
write the conclusion before the numbers exist.

Privacy: use synthetic forms here. Real forms hold CNIC and phone numbers and
must not be sent to a third-party API.

## 3. Explanation ratings (about 30 outputs, two raters)

```bash
# build blinded items: 15 forms x (LLM alone, LLM + RAG) = 30 items
python scripts/make_rating_set.py --dir data/synthetic/test --limit 15

# each rater, on their own machine/copy (never opens rating_key.json)
python -m streamlit run app/rate_explanations.py

# when both are done
python scripts/rating_report.py eval/rating/ratings_ayesha.json eval/rating/ratings_easha.json
```

- Raters score faithfulness, clarity and usefulness from 1 to 5 (guide in the sidebar).
  Scores save after every item and resume where you stopped.
- Do not share `rating_key.json` until both raters are finished, or the rating is not blind.
- `rating_report.py` prints and saves (`eval/rating/results.md`, `results.json`): mean
  score per condition, RAG better / tie / worse per form, and agreement between the raters
  (exact, within one point, Cohen's kappa, quadratic-weighted kappa with a Landis and Koch reading).
- Try the flow first with `--offline` (templated text, a TEST SET banner shows, no key, no OCR models).
  Never use offline ratings for results.
- Uses `formsense.llm.explain` when it exists; otherwise a small built-in Groq explainer.
  CNIC, phone and email are masked before any text is sent.

## 4. Tests

```bash
python -m pytest -m "not slow"                       # everything fast (about 2 minutes)
python -m pytest tests/test_vlm.py tests/test_rating.py tests/test_summary.py tests/test_apps.py
python -m pytest -m slow                             # real OCR models
```

## 5. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `AssertionError` in `faiss ... assert d == self.d` when searching the guidelines | The saved index was built with a different embedding model. Fixed: the retriever now records the model in the index and rebuilds by itself. If you still see it, delete the folder once: Windows `rmdir /s /q data\guidelines\.index`, Mac/Linux `rm -rf data/guidelines/.index`. It rebuilds on the next run (needs the MiniLM model, already cached on your machine). |
| `model_not_found` / "does not exist or you do not have access" from Groq | Groq retired the model. Run `python scripts/groq_models.py`, pick a current vision model, set `GROQ_VLM_MODEL` in `.env`. The error message in the app now says this and lists what your key can use. |
| Groq `429` / rate limit | Free-tier requests-per-minute cap. The VLM retries with backoff; raise `--delay` for long runs. Cached answers are never re-requested. |
| Many "unreadable" warnings | OCR is switched off in the sidebar. Turn "Run OCR" on. |

**For Easha:** `llama-3.3-70b-versatile`, the explainer model named in the project proposal, was also
retired by Groq (2026-08-16). Use a current text model (`qwen/qwen3.8-27b` or `openai/gpt-oss-120b`), keep
it configurable through `.env`, and use `formsense/groq_util.py` (`make_client()`, `chat()`): it handles
retries, rate limits and the retired-model error. Note that `gpt-oss-*` models spend part of
`max_completion_tokens` on reasoning, so leave generous headroom.
