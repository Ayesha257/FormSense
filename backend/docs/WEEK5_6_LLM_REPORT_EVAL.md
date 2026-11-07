# FormSense Weeks 5 and 6: LLM Explainer, Report, Evaluation

Owner: Easha. Folders: `formsense/llm/`, `formsense/report/`, `formsense/eval/`, `tests/`.

This continues `docs/WEEK3_4_NLP_AND_RULES.md`. It explains what was built in Weeks 5 and
6, how it works, why each choice was made, and how to run, test and extend it.

## 1. What was built, in plain words

Weeks 3 and 4 turned a form into fields and issues. Weeks 5 and 6 turn that into something
a person can read and something the team can measure:

1. **Week 5, explain the result (`llm/`).** Mask personal data, then ask an LLM to turn
   the issues into plain English, using LangChain's structured output so the reply is
   always a checkable shape, never free text that has to be parsed by hand.
2. **Week 6, report and evaluate (`report/`, `eval/`).** Package fields, issues and the
   explanation into the `AuditReport` from `schema.py`, write it as JSON and as a PDF, and
   measure the whole pipeline against ground truth with the metrics the brief asks for.

## 2. How the pieces connect

```
result = audit(fields)                      (Weeks 3-4)
        |
        v
explanation = explain_audit(result.fields, result.issues, template)     (Week 5)
        |                                    masks CNIC/phone/email first
        v
report = build_report(form_id, result.fields, result.issues, explanation)  (Week 6)
        |
        +--> write_json_report(report, path)
        +--> write_pdf_report(report, path, template)
        |
        v
evaluate_dataset(label_dir, system) -> metrics      (Week 6-7, separate from one form)
```

One command runs the first three steps:

```
python -m formsense.rules.audit data/synthetic/dev/labels/dev_0007.json \
  --template data/templates/admission_form.json --today 2026-11-06 \
  --report-out out/ --offline
```

Drop `--offline` once a Groq key is set, to get real LLM explanations instead of the
templated fallback.

## 3. Files

| File | What it does |
|---|---|
| `llm/pii_mask.py` | Masks CNIC, phone, email (always) plus any extra fields you name. |
| `llm/explanation_schema.py` | Pydantic shape of the LLM's structured reply. |
| `llm/prompts.py` | Builds the system and user prompts. |
| `llm/explainer.py` | `explain_audit()`: masks, prompts, calls a pluggable LLM client, checks for leaks. |
| `report/build.py` | `build_report()`: assembles `schema.AuditReport`. |
| `report/json_report.py` | Report to JSON string or file. |
| `report/pdf_report.py` | Report to PDF (ReportLab): summary, field checklist, issues table. |
| `eval/metrics.py` | CER, exact match accuracy, precision/recall/F1, confusion matrix. |
| `eval/harness.py` | Runs a system over a label dataset, aggregates metrics, compares systems. |

## 4. Week 5 in detail: `llm/`

### 4.1 PII masking (`pii_mask.py`)

The brief says CNIC, phone and email must be masked before the LLM call. This is the one
place that decides what "masked" means, so nothing downstream can forget.

* **Always masked, no way to turn it off:** `cnic`, `phone`, `email`. A caller can only
  add more field names to mask, never remove these three.
* **Shape preserving masking,** so the model can still reason about format problems
  without seeing the real value:

  | Field | Example | Masked |
  |---|---|---|
  | CNIC | `35202-1234567-1` | `35XXX-XXXXXXX-X` |
  | Phone | `0300-1234567` | `03XX-XXXXXXX` |
  | Email | `ali.raza@x.com` | `a***@x.com` |

  A value that does not match the expected shape (already broken, which is exactly the
  kind of value an issue might be about) becomes `[REDACTED CNIC]` and so on, since there
  is no safe partial form to show.
* A field already MISSING or UNREADABLE has no text, so nothing is masked or changed.
* `mask_text()` additionally scrubs any occurrence of a masked field's real value out of
  free text, for anything built from the fields before this point.
* `assert_no_pii()` is a last check, independent of whether masking ran correctly
  upstream: it raises if a real CNIC, phone or email value is still present in a string.
  `explain_audit()` calls it on the finished prompt, and again on the model's reply,
  before returning.

### 4.2 Explanation shape (`explanation_schema.py`)

```python
class ExplanationOutput(BaseModel):
    summary: str
    issues: list[IssueExplanation]       # issue_id + explanation, one per input issue
    overall_recommendation: str | None
```

This is separate from `schema.py`'s `AuditReport.explanation`, which is a single string:
`report/build.py` flattens the structured output into that string. Keeping the structured
shape separate means the LLM's reply can be validated field by field (did it answer every
issue, is `issue_id` one we actually sent) before it is flattened into text.

### 4.3 Prompting (`prompts.py`)

The system prompt tells the model: do not invent problems, do not repeat PII (refer to
"your CNIC" not the masked string either), do not mention rule ids or anything
implementation specific, name the actual field for every issue, and keep the tone like a
helpful clerk. The user prompt lists the notable fields (anything not OK, plus the value
of anything OK with a problem) and every issue in the system's own words, and asks the
model to translate those into plain English rather than copy them.

### 4.4 Calling the model (`explainer.py`)

```python
from formsense.llm.explainer import explain_audit

out = explain_audit(result.fields, result.issues, template)
```

The real call goes through an `LLMClient`: anything with `explain(system_prompt,
user_prompt) -> ExplanationOutput`. The default, `GroqClient`, wraps
`langchain_groq.ChatGroq(...).with_structured_output(ExplanationOutput)` exactly as the
brief specifies. It is built lazily: importing `explainer.py`, or even constructing
`GroqClient()`, never touches the network or requires `langchain_groq` to be installed
unless `.explain()` is actually called.

**Graceful degradation.** If no `GROQ_API_KEY` is set (checked at call time, or via
`GroqClient(api_key=...)`), `explain_audit` logs a warning and returns a plain templated
explanation instead of raising. Pass `offline=True` to skip the LLM on purpose, for a demo
or a batch run that should not depend on the network or a key. This means the rest of the
pipeline (report generation, the Streamlit app) keeps working even without a key
configured; only the real, model written explanations need it.

**Testing without the network.** Tests construct a small fake client:

```python
class FakeClient:
    def explain(self, system_prompt, user_prompt):
        return ExplanationOutput(summary="...", issues=[...], overall_recommendation=None)
```

and pass it as `client=FakeClient()`. This is how `test_llm_explainer.py` checks the
prompt never contains a real CNIC, phone or email across 100 randomly corrupted forms,
without ever calling Groq.

## 5. Week 6 in detail: `report/` and `eval/`

### 5.1 Report (`report/`)

`build_report(form_id, fields, issues, explanation, citations)` returns a plain
`schema.AuditReport`. It does no new judging; every decision was already made upstream.
`explanation_to_text()` flattens the structured LLM output into the single string
`AuditReport.explanation` expects: the summary, one line per issue, then the
recommendation.

`report_to_json()` / `write_json_report()` is a thin wrapper over Pydantic's own
serialisation.

`build_pdf()` / `write_pdf_report()` uses ReportLab to lay out: a title, the explanation
(if any), a field table (label, status, value or checked/unchecked), and an issues table
sorted by severity, colour coded. Giving it `template=` shows field labels instead of
internal names; without a template it falls back to raw names, so it still works on an
input with no template handy.

**PII note:** the PDF is not masked by this module. The applicant's own copy of their own
report should show their real CNIC. If a report is going anywhere other than the
applicant (a shared log, an email to a third party), mask it first with
`formsense.llm.pii_mask.mask_fields()` before calling `build_pdf`, the same way the LLM
prompt is masked.

### 5.2 Evaluation (`eval/`)

**Metrics (`metrics.py`),** matching the brief's list exactly:

| Brief's metric | Function |
|---|---|
| Field extraction: exact match accuracy, CER | `score_field_extraction` |
| Missing field detection: precision/recall/F1 | `score_missing_field_detection` |
| Inconsistency detection: precision/recall/F1 | `score_issue_detection` (and `types=` to filter to one `IssueType`, since format and selection issues are scored the identical way) |
| Missing vs unreadable: confusion matrix | `score_missing_vs_unreadable` |

`char_error_rate` is a plain Levenshtein distance divided by the ground truth length,
capped at 1.0. `score_issue_detection` matches issues by `(rule id, exact field set)`,
the same `issue_key` used elsewhere in the project, so an issue that found the right rule
but named the wrong fields is correctly scored as both a false positive and a false
negative, not a lucky match.

**Harness (`harness.py`).** A "system" is anything with `.fields` and `.issues` after
being called on a list of `FieldResult` (optionally a second `today` argument); `audit()`
already has this shape, so:

```python
from formsense.eval.harness import evaluate_dataset, full_pipeline_system, format_results

results = evaluate_dataset("data/synthetic/test/labels", full_pipeline_system(tpl))
print(format_results(results))
```

`evaluate_dataset` loads every label JSON in a directory, strips each field down to what a
real vision pipeline would hand over (`raw_text`, `ink_ratio`, `checked`; `value` and
`status` are cleared so the system is scored on its own extraction, not handed the
answer), runs the system, and aggregates every metric above across the whole dataset.

**Baseline versus full pipeline,** the brief's main result:

```python
from formsense.eval.harness import compare_systems, format_results

results = compare_systems("data/synthetic/test/labels", {
    "baseline": your_baseline_system,     # wrap Ayesha's formsense/baseline/ocr_regex.py the same way
    "full": full_pipeline_system(tpl),
})
print(format_results(results))
```

A baseline system only needs `.fields` and `.issues` after being called; it does not need
to import anything from this project. See section 7 for exactly what is still needed to
wire the real baseline in.

## 6. Decisions made for real world use

| Decision | Why |
|---|---|
| PII masking is mandatory and unconfigurable for CNIC, phone, email | The brief requires it; making it optional would make it easy to accidentally skip. |
| `assert_no_pii` runs on both the prompt and the reply | Masking upstream can be bypassed by a bug; the check is a second, independent gate right before anything leaves the process. |
| Structured output via Pydantic, not free text parsing | A free text reply would need fragile parsing; `with_structured_output` fails loudly instead of silently misparsing. |
| LLM client is pluggable and lazily constructed | Tests run with no network and no API key; the production code path is identical to what tests exercise, not a separate "real" implementation. |
| Offline fallback instead of raising | A missing key should degrade the explanation, not break the report, the demo, or a batch evaluation run. |
| `report/` does no new judging | Keeps the report a faithful rendering of what Weeks 3 to 5 already decided; easier to reason about and to test. |
| PDF is not masked by default | The applicant's own report should show their own real CNIC; masking is the caller's choice for a report that leaves the system. |
| `eval/` works on `.fields` / `.issues` duck typing, not a specific class | The baseline system (plain OCR + regex, a different module entirely) can be evaluated with the exact same harness and metrics as the full pipeline, which is the brief's whole comparison. |
| Issues are matched by rule id plus exact field set | A rule firing on the wrong fields is a real error and must count as one, not as a lucky match. |

## 7. What is needed from the team before this is fully wired

* **Groq API key.** Set `GROQ_API_KEY` in `.env` (your tree already has `.env.example`).
  Without it, `explain_audit` falls back to the templated explanation; nothing breaks, but
  the Week 5 evaluation ("LLM alone vs LLM + RAG", human rated) needs real model output.
* **RAG (`rag/`).** Per the ground rules, `rag/` is Ayesha's folder. `explain_audit` already
  accepts `guideline_context: Sequence[str]`, a list of short excerpts to include in the
  prompt. Whoever builds the FAISS retriever over Dania's guidelines doc just needs to
  pass its top results into that parameter; nothing in `llm/` needs to change.
* **Baseline wiring.** `formsense/baseline/ocr_regex.py` already exists in your tree. To
  compare it with `compare_systems`, wrap its output in anything with `.fields` and
  `.issues` matching `schema.FieldResult` / `schema.Issue` (even a tiny dataclass is
  enough). Send me `ocr_regex.py` and I will write that adapter.
* **Real OCR noise.** `evaluate_dataset`'s ground-truth stripping currently hands the
  system perfect `raw_text`. To evaluate against real OCR mistakes, either run Ayesha's
  actual pipeline on the synthetic images and build `LabelledForm`-shaped input from its
  output, or inject synthetic OCR noise into `raw_text` before scoring (not built yet,
  flagged as a gap below).

## 8. Testing

```
python -m pytest tests -q
```

218 tests pass (151 from Weeks 3 to 4, plus the ones below), 1 skipped (the generator
label comparison, same as before, only runs when `data/synthetic/*/labels` exist).

| Test file | What it proves |
|---|---|
| `test_pii_mask.py` | Every masker's exact output; CNIC/phone/email are always masked regardless of config; MISSING/UNREADABLE fields are skipped; `assert_no_pii` raises on a leak and passes on clean text. |
| `test_llm_explainer.py` | 100 randomly corrupted forms never leak a real CNIC, phone or email into the prompt; the client receives masked values but untouched names; offline mode never calls a client; a missing API key falls back instead of raising; a dict-shaped reply is still accepted. |
| `test_report.py` | JSON round trips through `AuditReport`; PDF output starts with `%PDF` and does not crash with zero issues or no template. |
| `test_eval_metrics.py` | CER on identical, substituted and empty strings; exact match and per-field breakdown; precision/recall/F1 including the zero-division NaN case; issue matching requires the exact field set; confusion matrix confusion and "other" cases. |
| `test_eval_harness.py` | The full pipeline recovers its own ground truth exactly (accuracy 1.0, F1 1.0) when run on its own labels; a deliberately empty "do nothing" system scores measurably worse, proving the metrics actually discriminate. |

## 9. Commands

```
pip install langchain langchain-groq reportlab pyyaml rapidfuzz dateparser numpy

# audit, explain (offline) and write a report in one call
python -m formsense.rules.audit data/synthetic/dev/labels/dev_0007.json \
  --template data/templates/admission_form.json --today 2026-11-06 \
  --report-out out/ --offline

# same, with a real Groq key set in the environment
python -m formsense.rules.audit data/synthetic/dev/labels/dev_0007.json \
  --template data/templates/admission_form.json --today 2026-11-06 --report-out out/
```

```python
# evaluate the full pipeline on the held-out test set
from formsense.eval.harness import evaluate_dataset, full_pipeline_system, format_results
from formsense.nlp.template_info import TemplateInfo

tpl = TemplateInfo.load("data/templates/admission_form.json")
results = evaluate_dataset("data/synthetic/test/labels", full_pipeline_system(tpl))
print(format_results(results))
```

## 10. Known limitations and open questions

* **No real LLM call has been tested here,** since this environment cannot reach Groq's
  API. The structured output path is exercised with a fake client that mimics the real
  interface; run it once against your real key before the Week 5 human rating exercise.
* **RAG is not built here.** `explain_audit` accepts guideline excerpts but does not fetch
  them; that is Ayesha's `rag/` module per the ground rules.
* **No synthetic OCR noise injector yet.** `evaluate_dataset` currently scores the
  language side against perfect OCR text. For an evaluation that reflects real OCR error,
  either run the real vision pipeline first, or we add a noise injector (character
  substitution plus a confidence drop) as a follow-up.
* **Baseline adapter not written,** pending `formsense/baseline/ocr_regex.py`'s real
  interface (see section 7).
* **PDF layout is plain,** by design, to keep ReportLab usage simple and robust; a visual
  pass (fonts, a logo, page breaks on a long issues list) can come later without changing
  the data it reads.
* **`overall_recommendation` fallback logic** (`_fallback_output`) just names the fields
  of the highest severity issue; it is a safety net, not a substitute for the real model's
  judgement.

## 11. What comes next

Week 7 is running the full evaluation (`eval/harness.py` is ready for this), the baseline
versus full pipeline comparison, and error analysis. Week 6's other half, Ayesha's RAG
module and Streamlit UI v1, slot into the `guideline_context` parameter and into calling
`formsense.rules.audit.audit()` plus `formsense.llm.explainer.explain_audit()` directly
from the UI.
