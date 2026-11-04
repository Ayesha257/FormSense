# FormSense: Multimodal Intelligent Form Auditor

NLP course project (Multimodal Document Assistant). Upload a filled university
admission form and FormSense checks whether it is complete, consistent and
valid, then explains the problems in plain English.

**Pipeline:** PDF/image → clean → align to template → OCR per field →
extract values → rules + cross-field checks → LLM explanation (+ RAG) → audit report

## Team & ownership

| Folder | Owner |
|---|---|
| `formsense/ingest`, `vision`, `baseline`, `rag`, `app/` | Ayesha |
| `formsense/generator`, `nlp`, `rules`, `llm`, `report`, `eval/` | Easha |
| `data/guidelines/`, real form collection, final slides | Dania |
| `formsense/schema.py` | Everyone (frozen after Week 1) |

## Setup (each person, once)

```bash
git clone https://github.com/<owner>/formsense.git
cd formsense
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Mac / Linux:    source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env
# paste your own Groq key into .env
python scripts/check_setup.py
streamlit run app/streamlit_app.py
```

## Git rules

1. Never push straight to `main`. Work on your own branch:
   `git checkout -b ayesha/vision` (or `easha/rules`, `dania/guidelines`)
2. Commit small and often: `git add . && git commit -m "Add deskew"`
3. Push your branch: `git push -u origin ayesha/vision`
4. Open a Pull Request on GitHub; one teammate glances at it, then merge.
5. Before starting work each day: `git checkout main && git pull`, then
   `git checkout <your-branch> && git merge main`
6. Changing `schema.py`? Tell the group first.

## Data rules

- `data/real/` is git-ignored: real forms contain people's CNIC and phone
  numbers. Share them through the team's private Google Drive folder.
- `data/synthetic/` is git-ignored: regenerate it with the generator script.
- `.env` is git-ignored: never commit API keys.

## Template

`data/templates/admission_form.json` lists every field. After we design the
form, put the blank form image at `data/templates/admission_form_blank.png`
and fill in each field's `bbox` as `[x, y, width, height]` in pixels of that image.
