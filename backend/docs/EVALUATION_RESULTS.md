# FormSense: Evaluation Results

8 November 2026. NLP course project.
Test set: 60 held-out synthetic admission forms. Real-form results are pending (section 7).

## 1. Summary

On 60 held-out synthetic forms, the full FormSense pipeline (reading from the page image) extracts 81.8% of fields exactly, against 40.5% for the plain OCR and regex baseline, and its character error rate is 0.082 against 0.448. It finds empty fields with F1 0.96 (baseline 0.04) and form issues with F1 0.45 (baseline 0.09).

When the language side (field extraction and the rule engine) is given perfect text instead of OCR output, issue F1 rises to 0.94. The gap between 0.45 and 0.94 therefore comes from reading handwriting-style text, not from the rules.

The main weaknesses of the full pipeline are OCR errors on numeric fields and a large number of false "unreadable" flags, which together keep issue precision at 0.31.

Explanations that use retrieved guideline text (LLM + RAG) were preferred over explanations without it on usefulness in 21 of 30 judged comparisons, and the other 9 were ties. These ratings come from two LLM judges, not from human raters.

| Metric | Baseline | Full (images) | Full (ground-truth text) |
|---|---|---|---|
| Field exact-match accuracy | 0.405 | 0.818 | 0.998 |
| Character error rate | 0.448 | 0.082 | 0.001 |
| Missing-field P / R / F1 | 0.02 / 0.92 / 0.04 | 0.92 / 1.00 / 0.96 | 1.00 / 1.00 / 1.00 |
| Issue-detection P / R / F1 | 0.05 / 0.49 / 0.09 | 0.31 / 0.87 / 0.45 | 1.00 / 0.88 / 0.94 |

The right-hand column is an upper bound for the language side. It is not a deployable system, because it is given the correct text.

## 2. How the evaluation was done

### 2.1 Test data

The test set is 60 synthetic admission forms in `data/synthetic/test/`, kept out of development. Each form is a page image (PNG or JPEG) produced by the form generator, which fills the template with fake data in handwriting-style fonts and injects errors on purpose (blank signature, truncated CNIC, inconsistent marks, two programmes ticked, and so on). Because the generator knows what it injected, every form has a ground-truth label file in `data/synthetic/test/labels/` with the correct value and status of each field and the list of true issues.

- Field instances: 60 forms x 23 fields = 1,380. Twelve fields are empty by design, so 1,368 fields are scored for extraction accuracy and character error rate.
- True issues: 75 in total (12 missing, 7 unreadable, 24 format, 27 inconsistency, 5 selection).

### 2.2 Systems compared

1. **Baseline.** Full-page OCR (PaddleOCR), text lines rebuilt in reading order, each field's value taken from the text after its printed label, with a regex fallback. It has no layout model and no image analysis. It reports every signature as present and guesses ticked boxes from characters before an option label.
2. **Full (images).** The FormSense pipeline: clean the image (deskew, denoise), align it to the blank template, crop each field, detect ink for checkboxes, signatures and empty fields, run PaddleOCR on text fields with a TrOCR handwriting fallback, then extract and normalise values.
3. **Full (ground-truth text).** The same extraction and rule code, fed the correct text and ink readings from the label files instead of OCR output. This isolates the language side from the vision side.

All three systems pass through the same rule engine (required fields, format checks, cross-field checks), so differences in issue detection come from the field readings and not from different rules.

### 2.3 Metrics

- **Exact-match accuracy:** share of scored fields whose extracted value equals the ground-truth value.
- **Character error rate (CER):** character-level edit distance between extracted and true text, averaged over fields.
- **Missing-field detection:** precision, recall and F1, with "field is empty in the ground truth" as the positive class.
- **Issue detection:** precision, recall and F1 of the predicted issues against the ground-truth issue list, overall and per issue type.
- **Missing vs unreadable:** confusion matrix of true status against predicted status. A field with no ink is missing; a field with ink but low OCR confidence is unreadable.

### 2.4 Procedure

`formsense/eval/run_comparison.py` runs each system on every test form, passes the fields through the rule engine, scores the result against the labels, and saves each form's output. `formsense/eval/error_analysis.py` then sorts every wrong field into a cause. The run was done on a laptop CPU. The baseline took roughly 70 to 150 seconds per form (full-page OCR) and the full pipeline about 16 seconds per form.

## 3. Results

### 3.1 Overall

See the table in section 1. Raw issue counts (true positives / false positives / false negatives): baseline 37 / 748 / 38, full (images) 65 / 148 / 10, full (ground-truth text) 66 / 0 / 9.

### 3.2 By issue type

Precision / recall / F1. "n/a" means the value is undefined because there were no predictions or the denominator was zero.

| Issue type (true count) | Baseline | Full (images) | Full (ground-truth text) |
|---|---|---|---|
| Missing (12) | 0.02 / 0.92 / 0.04 | 0.92 / 1.00 / 0.96 | 1.00 / 1.00 / 1.00 |
| Unreadable (7) | 0.00 / 0.00 / n/a | 0.06 / 1.00 / 0.11 | n/a / 0.00 / n/a |
| Format (24) | 0.33 / 0.54 / 0.41 | 0.47 / 0.75 / 0.58 | 1.00 / 0.92 / 0.96 |
| Inconsistency (27) | 0.85 / 0.41 / 0.55 | 0.58 / 0.85 / 0.69 | 1.00 / 1.00 / 1.00 |
| Selection (5) | 0.02 / 0.40 / 0.03 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| All (75) | 0.05 / 0.49 / 0.09 | 0.31 / 0.87 / 0.45 | 1.00 / 0.88 / 0.94 |

Two things follow from the right-hand column. The 9 issues missed with perfect text are the 7 unreadable issues, which cannot be detected without an OCR confidence score, and 2 format issues. So the language side is close to complete on everything it can see.

The unreadable row for the full pipeline shows where precision is lost: 110 of its 148 false positives are false "unreadable" issues. Computed from the table above, with the unreadable type left out, the full pipeline has 58 true positives, 38 false positives and 10 false negatives, which is precision 0.60, recall 0.85 and F1 0.71.

### 3.3 Missing vs unreadable (synthetic forms)

Rows are the true status, columns the predicted status.

| | Full: missing | Full: unreadable |
|---|---|---|
| True missing (12) | 12 | 0 |
| True unreadable (7) | 0 | 7 |

The full pipeline separates the two cases without error on the synthetic forms. The baseline labelled 11 of the 12 empty fields as missing, but it also labelled 4 of the 7 unreadable ones as missing and put the rest in neither class, and it flagged nearly every field as missing (precision 0.02). The ground-truth-text system cannot report "unreadable" at all, because it has no OCR confidence. This matrix is to be repeated on real forms (section 7).

### 3.4 Per-field extraction

Exact-match accuracy and CER for the two image-based systems. Fields with no ground-truth value on a form are left out of that field's count (n = 59 for father's name, matric percentage and matric year; n = 51 for signature).

| Field | Baseline acc | Baseline CER | Full acc | Full CER |
|---|---|---|---|---|
| full_name | 0.000 | 0.310 | 0.733 | 0.103 |
| father_name | 0.000 | 0.312 | 0.644 | 0.099 |
| cnic | 0.700 | 0.231 | 0.817 | 0.088 |
| date_of_birth | 0.783 | 0.162 | 0.817 | 0.103 |
| email | 0.467 | 0.184 | 0.600 | 0.114 |
| phone | 0.667 | 0.290 | 0.750 | 0.150 |
| address | 0.000 | 0.277 | 0.600 | 0.042 |
| matric_year | 0.000 | 1.000 | 0.864 | 0.072 |
| matric_total | 0.000 | 1.000 | 0.867 | 0.078 |
| matric_obtained | 0.000 | 1.000 | 0.800 | 0.144 |
| matric_percentage | 0.000 | 1.000 | 0.746 | 0.128 |
| inter_year | 0.000 | 1.000 | 0.867 | 0.063 |
| inter_total | 0.000 | 1.000 | 0.733 | 0.117 |
| inter_obtained | 0.000 | 1.000 | 0.767 | 0.192 |
| inter_percentage | 0.000 | 1.000 | 0.717 | 0.154 |
| declaration_name | 0.000 | 0.299 | 0.733 | 0.061 |
| declaration_date | 0.767 | 0.193 | 0.783 | 0.155 |

The baseline never extracts any of the academic-marks fields (CER 1.0 on all eight). Their values sit in a table layout and are not on the same line as a label, which is the case the baseline's label-then-value rule cannot handle. The full pipeline reads them by position.

The gender, programme and signature fields score 1.000 for every system in this table. That figure is not evidence of tick or signature detection (see section 6, point 1); section 3.5 uses error analysis and the selection-issue scores instead.

### 3.5 Error analysis

Every wrong field was assigned to one cause.

| Cause | Meaning | Baseline | Full (images) |
|---|---|---|---|
| false_missing | value present, reported missing | 537 | 0 |
| ocr_text_error | text differs clearly from the truth | 86 | 133 |
| checkbox_wrong | tick state wrong | 122 | 0 |
| flagged_unreadable | readable field flagged unreadable* | 2 | 53 |
| signature_missed | blank signature reported as signed | 9 | 0 |
| signature_false_alarm | signed form reported as blank | 0 | 1 |
| false_present | empty field reported as filled | 1 | 0 |

\* This count also includes fields that are genuinely unreadable in the ground truth (7 in the test set).

Counts here are per field. The issue metrics in section 3.2 are per issue, so the two sets of numbers are not directly comparable.

**Baseline.** Most errors are fields reported missing when a value is there (537): when a label is not matched or its value is not on the same line, the baseline gives up. It gets 122 tick states wrong, which produces 118 false "selection" issues. It treats all 9 blank signatures as signed.

**Full pipeline.** No field is wrongly reported missing and no tick state is wrong. The remaining errors are OCR errors and unreadable flags:

- OCR errors are concentrated in numeric fields: inter_total (15), matric_percentage (13), inter_percentage (13), cnic (10), inter_obtained (9), together with father_name (9). A single misread digit is enough to break the format rules and the percentage-equals-obtained-over-total rule, which accounts for most of the false format (20) and inconsistency (17) issues.
- False unreadable flags are concentrated in phone (8) and full_name (5). The confidence threshold for "unreadable" is 0.60, and these results suggest it is too strict for handwriting-style text.
- Errors cluster in a few forms: test_0050 and test_0030 (12 OCR errors each) and test_0021 (11 unreadable flags), rather than being spread evenly.

### 3.6 Effect of the vision update

The full pipeline was run twice on the same 60 forms, before and after the updated vision code (alignment and field detection) was merged from the main branch.

| Full (images) | Before | After |
|---|---|---|
| Field exact-match accuracy | 0.814 | 0.818 |
| CER | 0.101 | 0.082 |
| Missing-field P / R / F1 | 0.16 / 1.00 / 0.27 | 0.92 / 1.00 / 0.96 |
| Issue P / R / F1 | 0.27 / 0.85 / 0.41 | 0.31 / 0.87 / 0.45 |
| false_missing fields | 63 | 0 |
| checkbox_wrong fields | 5 | 0 |

The update removed the false "missing" reports. Fields that used to be reported empty are now read, and some of them are misread or flagged unreadable, which is consistent with the rise in OCR errors (103 to 133) and unreadable flags (28 to 53). Baseline numbers moved by 0.01 or less between the two runs.

## 4. Explanation quality

### 4.1 Setup

For 15 test forms with at least one issue, two explanations were generated: one from the issues and form facts alone (LLM alone), and one that also receives the matching guideline passages retrieved from the guidelines document (LLM + RAG). That gives 30 explanations. Personal data (CNIC, phone, email) is masked before any text goes to the model. The explainer model is set in `.env` (default `qwen/qwen3.8-27b`).

The brief asks for two human raters and Cohen's kappa. For lack of time, two LLM judges (`openai/gpt-oss-120b` and `openai/gpt-oss-20b`) were used instead. These are not human ratings. The judges never see which explanation came from which condition, and neither judge wrote the explanations.

### 4.2 Absolute scoring (1 to 5)

Each explanation was scored for faithfulness, clarity and usefulness. Both conditions averaged 4.99 out of 5, and the two judges agreed on 98% of ratings. Cohen's kappa came out at 0.00, which is a side effect of almost every rating being 5, not a sign of disagreement. This scoring could not separate the two conditions.

### 4.3 Pairwise comparison

For each form the judge saw both explanations side by side and picked the better one per criterion (or a tie). Each comparison was run twice with the order swapped, and a pick that changed with the order was counted as a tie.

Counts over 15 forms and 2 judges (30 comparisons per criterion):

| Criterion | RAG better | Tie | LLM alone better |
|---|---|---|---|
| Faithfulness | 2 | 27 | 1 |
| Clarity | 11 | 19 | 0 |
| Usefulness | 21 | 9 | 0 |

Reliability: the order of presentation changed the pick in 16% (judge A) and 20% (judge B) of comparisons. The two judges agreed exactly on 76% of 45 comparisons (Cohen's kappa 0.51, moderate).

RAG explanations were never judged less useful or less clear, and faithfulness was almost always equal. The usefulness gain is visible in the text: the RAG version states the exact required format (for example the CNIC pattern), where the version without guidelines says only that the format is wrong.

## 5. Discussion

1. Layout and image evidence matter. The baseline fails on any field whose value is not next to its label, and it cannot see ticks or signatures. The full pipeline fixes both, which is the main result the project set out to show.
2. The rule engine is not the bottleneck. With correct text it finds every issue it can see. What limits the full pipeline is the OCR reading of handwriting-style text, particularly digits.
3. The next improvement is in vision, not in language: a calibrated "unreadable" threshold (110 false unreadable issues is the largest single source of false positives) and better digit recognition for the marks and ID fields.
4. Guideline retrieval improves the usefulness of explanations without making them less faithful, on this small set.

## 6. Limitations and open points

1. **Tick and signature rows in the per-field table are not informative.** All systems score 1.000 on the gender, programme and signature fields, yet the error analysis finds 122 wrong baseline tick states. The exact-match score for these fields evidently does not measure tick state, so use the error analysis and the selection-issue scores instead. The cause of this mismatch has not been traced.
2. **Baseline missing-field recall of 0.92 is not credible as signature detection.** The saved baseline output marks all 9 blank signatures as signed, but the scored recall is 0.92. The score is computed after the rule engine's extraction step, which may re-assign status. The precision of 0.02 shows the baseline reports nearly every field as missing, so its recall mostly reflects that. This needs checking before the baseline's signature behaviour is quoted.
3. **Synthetic data only.** Handwriting is a font, not human handwriting, and the generator that produced the test forms also produced the development forms (the test forms were held out, but the style is the same). Results on real photographed forms may be lower.
4. **Explanation ratings are from LLM judges**, from the same model family, on 15 forms. Comparisons within a form are not independent. The ratings support a direction (RAG is preferred on usefulness), not a precise percentage.
5. **The language-side upper bound** uses perfect text and is not a system anyone could deploy.
6. Issue matching rules and the exact-match normalisation follow the evaluation harness (`formsense/eval/metrics.py`) and are not restated here.

## 7. Real-form evaluation (pending)

About 15 filled admission forms are to be collected and photographed, and their mistakes recorded by hand. Real forms contain personal data, so they will not be sent to any third-party API. When they are available, both systems will be run on them with the same procedure and the same metrics, and this section will be filled in.

| Metric | Baseline | Full (images) |
|---|---|---|
| Forms | not run | not run |
| Field exact-match accuracy | not run | not run |
| CER | not run | not run |
| Missing-field P / R / F1 | not run | not run |
| Issue-detection P / R / F1 | not run | not run |
| Missing vs unreadable (confusion matrix) | not run | not run |

## 8. Files

| Path | Contents |
|---|---|
| `data/eval_results/results_table.txt` | Overall comparison table |
| `data/eval_results/results.json` | All metrics, including per field, per issue type and the confusion matrices |
| `data/eval_results/baseline_<form>.json`, `full_<form>.json` | Each system's fields for each form |
| `data/eval_results/error_analysis.csv` | One row per wrong field, with its cause |
| `data/eval_results/failed_forms.txt` | Forms on which a system crashed |
| `data/eval_results_v1/` | Results from before the vision update (section 3.6) |
| `eval/rating/` | Explanation set, judge scores and results (`llm_judge_results.md`, `llm_judge_pairwise_results.md`) |

To reproduce, run from the `backend` folder:

```
python -u -m formsense.eval.run_comparison
python -m formsense.eval.error_analysis
python scripts/make_rating_set.py --dir data/synthetic/test --limit 15 --explainer formsense.llm.explainer:explain
python scripts/llm_judge.py
python scripts/llm_judge_pairwise.py
```