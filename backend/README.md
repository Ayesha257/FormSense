# FormSense: Multimodal Intelligent Form Auditor

Upload a filled university admission form (scan or phone photo) and FormSense
checks whether it is complete, consistent and valid, then explains any
problems in plain English.

**Pipeline:** PDF/image → clean → align to template → OCR per field →
extract values → rules + cross-field checks → LLM explanation (+ RAG) → audit report

## Setup

```bash
git clone <repo-url>
cd FormSense/backend
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Mac / Linux:    source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env
# paste your Groq API key into .env (https://console.groq.com/keys)
python scripts/check_setup.py
```

Run every command from the `backend` folder.

## Usage

```bash
# Synthetic forms with ground truth (see formsense/generator/generate.py)
python -m formsense.generator.generate --count 80 --split dev  --seed 1      --out data/synthetic/dev
python -m formsense.generator.generate --count 40 --split test --seed 100000 --out data/synthetic/test

# Demo app: vision -> rules -> guideline for every issue
python -m streamlit run app/streamlit_app.py

# Vision step on one filled form
python scripts/run_vision.py path/to/filled_form.jpg

# Detector accuracy and best thresholds (tune on dev, report on test)
python scripts/eval_detectors.py --dir data/synthetic/dev

# FormSense vs. the OCR + regex baseline, fields and issues end to end
python scripts/compare_vision_baseline.py --dir data/synthetic/test --limit 20

# Guidelines retrieval (RAG)
python scripts/rag.py build
python scripts/rag.py ask "Can my mother sign the form for me?"
python scripts/rag.py eval

# Rules on a JSON list of field readings (or a generator label file)
python -m formsense.rules.audit path/to/fields.json

# Small demo form (independent of the admission form)
python scripts/make_demo.py
python scripts/run_vision.py data/templates/demo/samples/complete.jpg --demo

# Tests (fast ones only / everything incl. real OCR)
python -m pytest -m "not slow"
python -m pytest
```

In code, the whole chain is one call:

```python
from formsense.pipeline import FormSense
result = FormSense().run(page_image)      # result.issues, result.fields, result.citations
report = result.report()                  # AuditReport for the LLM explainer / report
```

## Project structure

```
backend/
  app/                  Streamlit demo
  formsense/
    schema.py           shared data models
    ingest/             PDF/image loading, denoise, deskew
    vision/             template alignment, field cropping, ink detection, OCR
    baseline/           OCR + regex baseline (plain OCR, no layout)
    generator/          synthetic form generator
    nlp/                field mapping, value extraction
    rules/              validation rules
    llm/                LLM explanations
    rag/                guideline retrieval (LangChain + FAISS, hybrid search)
    report/             JSON / PDF audit report
    scoring.py          compare predicted fields and issues with ground truth
    pipeline.py         vision -> rules -> guidelines in one call
  data/
    templates/          blank form image + field boxes (JSON)
    synthetic/          generated forms (not committed)
    real/               real filled forms (not committed: personal data)
    guidelines/         form-filling guidelines for RAG
  eval/                 evaluation scripts and results
  scripts/              setup check and helper tools
  tests/
```

## Template

`data/templates/admission_form.json` lists every field and its box, and
`data/templates/admission_form_blank.png` is the blank form used for alignment
(both made by `python -m formsense.generator.layout`). For a different paper
form, `python scripts/mark_fields.py` lets you draw the boxes by hand.

## Guidelines (RAG)

Put the form-filling guidelines as `.md` or `.txt` in `data/guidelines/`. The
search index rebuilds itself when a file changes. When the guidelines change,
update the questions in `data/guidelines/eval_queries.json` and the section names
in `scripts/rag.py` (FIELD_SECTION, RULE_SECTION).

## Notes

- `data/real/`, `data/synthetic/` and `.env` are git-ignored. Never commit
  real forms (they contain CNIC and phone numbers) or API keys.
