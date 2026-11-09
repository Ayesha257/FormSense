# Week 8: integration, real evaluation, diagnostics

Run every command from `backend/`.

## 1. What is wired together now

```
page image -> vision (align, ink, OCR) -> rules (extract + validate) -> guideline per issue (RAG)
           -> explanation (Easha's llm/, guidelines as context) -> report (JSON, PDF)
```

`formsense/explain.py` is the one place that joins the sides: `explain_report(report, use_rag=...)`,
`build_full_report(result, explanation)`, `pdf_bytes(report)`. The app, the rating-set builder and the
doctor all use it, so there is one code path to test.

Groq retired `llama-3.3-70b-versatile` (2026-08-16) and Llama 4 Scout (2026-07-17). Easha's explainer already
moved to `qwen/qwen3.8-27b` (override: `GROQ_LLM_MODEL` in `.env`) and asks for plain JSON, because Groq's tool
calling failed on some models. `explain_report` adds one thing on top: if the model is ever retired again, it
raises `LLMModelUnavailable` with what to change, instead of a raw API error.

## 2. The real evaluation (use this for the results slide)

Easha's `formsense/eval/run_comparison.py` runs the real OCR on the 60 test forms and saves each form's readings in
`data/eval_results/`; her `docs/EVALUATION_RESULTS.md` reports the numbers. `scripts/evaluate_all.py` scores the same
readings with the same metrics, and adds what that run does not show: where each error comes from, a text-only
accuracy, and the cutoff sweep. It can also run OCR itself.

```bash
# score the readings Easha already saved (no OCR, takes seconds)
python scripts/evaluate_all.py --dir data/synthetic/test --readings data/eval_results --systems formsense,baseline

python scripts/evaluate_all.py --dir data/synthetic/test --limit 20             # run OCR yourself (slow), quick look
python scripts/evaluate_all.py --systems formsense --reuse                      # re-score readings this script saved
python scripts/evaluate_all.py --systems formsense,baseline,vlm                 # also the Groq VLM (needs a key)
python scripts/evaluate_all.py --dry-run --limit 10                             # tests the chain; SIMULATED OCR
```

Output in `outputs/evaluation/`: `summary.md` (paste into slides), `summary.json`, `field_errors_<system>.csv`,
`issue_errors_<system>.csv`, `predictions/`.

**Labels must match the images.** The generator makes dates relative to the day it runs, so labels regenerated on
another day have different dates and every date field then scores about 0 (a harmless mismatch, but confusing).
Easha's test set was generated for 2026-11-06; to regenerate identical labels use
`python -m formsense.generator.generate --count 60 --split test --seed 100000 --today 2026-11-06 --out data/synthetic/test`.
Better: commit `data/synthetic/test/labels/` (small JSON, synthetic, no personal data) so everyone scores the same truth.

**Why the ceiling row exists.** `evaluate_dataset` with its default input hands the rules PERFECT text from the labels.
That is a ceiling for the language side, not the system's accuracy.

**Reading the error breakdown.** Every field gets one outcome (ok, misread, lost_unreadable, false_content, ...).
Every wrong issue is traced: if a field it is about was read wrongly, the cause is `field_read_error` (fix OCR/vision);
otherwise `rule_logic` (fix the rule). That tells you where to spend time.

### What it shows on the real readings (60 test forms)

| System | Field accuracy (brief's metric) | Text-field accuracy | Issue P / R / F1 |
| --- | --- | --- | --- |
| Ceiling (perfect text) | 0.998 | 0.997 | 1.00 / 0.88 / 0.94 |
| FormSense (images), cutoff 0.60 | 0.818 | 0.755 | 0.36 / 0.87 / 0.51 |
| Baseline (OCR + regex) | 0.405 | 0.200 | 0.05 / 0.51 / 0.09 |

* **The rules are not the problem.** 126 of 127 wrong FormSense issues trace to a field that OCR read wrongly.
* **The brief's field accuracy is inflated.** It compares text, and tick boxes and the signature have none (empty equals
  empty), so those 351 of 1,368 scored fields always count as correct. Written fields only: 75.5% (FormSense) and 20.0%
  (baseline). Quote the text-field figure, and say how ticks and signatures were scored (error breakdown: 0 tick errors, 1
  missed signature out of 360 tick and signature checks).
* **Biggest error sources:** 46 false `unreadable_field` warnings (see section 3), then digit misreads read with HIGH
  confidence (1100 as 1700, 2021 as 2027, 69.91 as 69.97, confidence 0.84 to 0.99), which no confidence cutoff can catch and which
  cause the `percentage_mismatch` (16) and `invalid_number` (16) false alarms.
* **Next improvement is vision/OCR, specifically digits:** digit-only recognition for the numeric fields, or reading
  twice and keeping the answer both engines agree on. A post-OCR repair using `percentage = obtained / total` could also
  fix single misread digits; measure it with `--readings` before claiming it.

## 3. The "unreadable" cutoff

A text box with ink and OCR confidence below a cutoff is UNREADABLE and raises an `unreadable_field` warning.
That cutoff existed twice (0.60 in vision, 0.80 in `nlp/extract.py`, which runs last and so decided), so the system
effectively ran at **0.80** (the "0.60" in `docs/EVALUATION_RESULTS.md` section 3.5 is the vision-step value that
`audit()` overrides; the 110 false "unreadable" issues in section 3.2 are the 0.80 behaviour). It is now one constant,
`formsense/thresholds.py`.

Sweep of the cutoff on the saved real readings (`--readings ... --sweep-threshold`):

| Cutoff | Issue P | Issue R | Issue F1 | False "unreadable" | Truly unreadable found |
| --- | --- | --- | --- | --- | --- |
| 0.80 (old effective) | 0.31 | 0.87 | 0.45 | 110 | 7 of 7 |
| **0.60 (new default)** | 0.36 | 0.87 | 0.51 | 46 | 6 of 7 |
| 0.40 | 0.37 | 0.85 | 0.52 | 6 | 4 of 7 |

0.60 removes most false alarms and still finds 6 of the 7 truly unreadable fields. **It was read off the test readings,
so treat it as provisional**: a two-fold check inside the test set (choose on half, score on the other half) gave F1 0.49
against 0.41 for the old cutoff on one fold and 0.49 against 0.50 on the other, so the gain is real but modest and
noisy. Confirm it on the dev split before quoting it:

```bash
python -m formsense.generator.generate --count 60 --split dev --seed 1 --today 2026-11-06 --out data/synthetic/dev
python scripts/evaluate_all.py --dir data/synthetic/dev --systems formsense --sweep-threshold   # runs OCR, ~16 s per form
```

If dev disagrees, change `OCR_UNREADABLE_BELOW` in `formsense/thresholds.py`, re-run `run_comparison`, and report test once.
No cutoff fixes this completely: confidence barely separates smudged fields from ordinary handwriting-font fields.

## 4. The doctor

```bash
python scripts/doctor.py                       # offline: setup, guidelines, RAG index and quality, RAG -> LLM wiring
python scripts/doctor.py --online --ocr        # also Groq text and vision models, and a PaddleOCR read
python scripts/doctor.py --fix                 # rebuild a broken guidelines index
```

PASS / WARN / FAIL per check, with what to do. Run it before every demo. If someone says "RAG is not working",
send the output of this command: it shows whether the index, the retrieval quality or the hand-over to the LLM is
the problem.

## 5. Known limits

- The guidelines in `data/guidelines/` are the SAMPLE draft until Dania's final document replaces them. After that,
  update `data/guidelines/eval_queries.json` and run `python scripts/rag.py eval`.
- Real-form scoring needs a labelled spreadsheet turned into the label format; `evaluate_all.py` reads generator labels.
- Numbers from `--dry-run` describe the simulated noise, not the system.
