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
# Try everything on a demo form (before the real form exists)
python scripts/make_demo.py
python scripts/run_vision.py data/templates/demo/samples/complete.jpg --demo

# Mark field boxes on the real blank template (once, after the form is designed)
python scripts/mark_fields.py

# Run the vision step on one filled form
python scripts/run_vision.py path/to/filled_form.jpg

# Demo app
streamlit run app/streamlit_app.py

# Tests
pytest
```

## Project structure

```
backend/
  app/                  Streamlit demo
  formsense/
    schema.py           shared data models
    ingest/             PDF/image loading, denoise, deskew
    vision/             template alignment, field cropping, ink detection, OCR
    baseline/           OCR + regex baseline
    generator/          synthetic form generator
    nlp/                field mapping, value extraction
    rules/              validation rules
    llm/                LLM explanations
    rag/                guideline retrieval
    report/             JSON / PDF audit report
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

`data/templates/admission_form.json` lists every field. Put the blank form
image at `data/templates/admission_form_blank.png`, then run
`python scripts/mark_fields.py` to draw each field's box.

## Notes

- `data/real/`, `data/synthetic/` and `.env` are git-ignored. Never commit
  real forms (they contain CNIC and phone numbers) or API keys.
