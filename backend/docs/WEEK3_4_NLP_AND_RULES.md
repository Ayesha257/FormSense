# FormSense Weeks 3 and 4: Language and Reasoning

Owner: Easha. Folders: `formsense/nlp/`, `formsense/rules/`, `tests/`.

This document explains what was built in Weeks 3 and 4, how it works, why each design
choice was made, and how to run, test and extend it. It is written so that a teammate who
has never opened the code can understand the system.

## 1. What was built, in plain words

The vision side (Ayesha) looks at the form image and, for every field box, produces a
reading: the text OCR found, how confident OCR was, how much ink is in the box, and
whether a checkbox is ticked. That reading is raw and unjudged.

Weeks 3 and 4 turn those readings into a verdict:

1. **Week 3, understand the values (`nlp/`).** Decide for each field whether it is
   MISSING, UNREADABLE, INVALID or OK, and turn the text into a clean value (a date
   becomes `2005-03-14`, a percentage `85%` becomes `85`). Also map printed labels such as
   "D.O.B" to the canonical field name `date_of_birth`.
2. **Week 4, judge the form (`rules/`).** Run 19 rules, all defined in one YAML file,
   over the cleaned fields and return a list of `Issue` objects: required fields, format
   checks, checkbox groups and cross field checks such as "CNIC last digit does not match
   the ticked gender".

Both parts use only the shared contract in `schema.py`. Nothing in `schema.py` was changed.

## 2. How the pieces connect

```
Vision output (list of FieldResult: raw_text, ocr_confidence, ink_ratio, checked)
        |
        v
nlp/extract.py        status + normalised value per field      (Week 3)
        |
        v
rules/engine.py       Issue objects from rules.yaml            (Week 4)
        |
        v
rules/audit.py        one call that does both and marks INVALID fields
        |
        v
LLM explainer (Week 5), report generator (Week 6), evaluation (Week 6 to 7)
```

The single function the rest of the project needs:

```python
from formsense.rules.audit import audit

result = audit(field_results, today=date.today())
result.fields   # FieldResult list with status and value filled in
result.issues   # list[Issue]
result.notes    # {field name: ["human readable extraction notes"]}
```

If the fields are generator ground truth (statuses already final), call
`audit(fields, extract=False)`.

## 3. Files

| File | What it does |
|---|---|
| `nlp/template_info.py` | Reads the template JSON once and answers "what type is this field, is it required, which checkbox group is it in". No other module hardcodes field lists. |
| `nlp/normalize.py` | One normaliser per value type: CNIC, phone, email, date, number, year, name, text. Also `names_match` for fuzzy name comparison. |
| `nlp/extract.py` | Decides MISSING / UNREADABLE / INVALID / OK and fills `value`. |
| `nlp/field_mapper.py` | Maps a printed label to a canonical field name in three stages: exact, fuzzy, MiniLM embeddings. |
| `nlp/label_aliases.yaml` | Synonyms for labels, for example "D.O.B" and "Mobile No.". |
| `nlp/calibrate.py` | Command line tool that suggests ink thresholds from your labelled forms. |
| `rules/rules.yaml` | All 19 rules as data. |
| `rules/engine.py` | Loads the YAML, validates it, runs the rules, returns `Issue` objects. |
| `rules/audit.py` | `audit()` glue function and a small command line. |
| `tests/` | 150 plus tests (see section 8). |

`rules/codes.py` and `rules/reference_validator.py` already existed from the generator work
and were not modified.

## 4. Week 3 in detail: `nlp/`

### 4.1 Normalisers (`normalize.py`)

Every normaliser returns a `Norm(value, ok, reason, repaired)`. None of them ever raises
an exception. A failure is `ok=False` with a short reason such as `cnic_format`.

**Design rule 1: strict by default.** A CNIC with a missing dash is a genuine applicant
mistake. If the normaliser quietly repaired it, the audit would hide the very error it
exists to find. So by default nothing is fixed.

**Design rule 2: repair only when OCR is unsure.** The extractor passes `repair=True`
only when OCR confidence is below 0.90. In that case it fixes common OCR confusions:
`O` for `0`, `l` or `I` for `1`, `S` for `5`, `B` for `8`, missing dashes in a CNIC or
phone number, a `+92` or `0092` phone prefix, and `(at)` or `,com` in emails. The result is
flagged `repaired=True` and appears in the notes.

The reasoning: if OCR is 98 percent sure it read the letter `O` inside a CNIC, then the
applicant really wrote an `O`, and that is a real error. If OCR is 85 percent sure, a slip is
more likely.

| Type | Strict rule | Examples of what repair can fix |
|---|---|---|
| CNIC | `#####-#######-#` | `3520212345671` becomes `35202-1234567-1`; `35202-12345O7-1` becomes `35202-1234507-1`. A 12 or 14 digit string is never "fixed". |
| Phone | `03XX-XXXXXXX` | `+92 300 1234567` becomes `0300-1234567` |
| Email | standard pattern | `ali (at) gmail,com` becomes `ali@gmail.com` |
| Date | day first formats: `14/03/2005`, `14-03-2005`, `14 Mar 2005`, `14.03.2005`, `14 March 2005`, ISO. Falls back to `dateparser` (day first, strict, needs day, month and year) for styles like `5th March 2005`. | `O5/O3/2OO5` becomes `2005-03-05`. `31/02/2005` is always rejected. |
| Number | digits, optional decimals, optional `%`, thousands commas | `8O` becomes `80`. Words, negatives, `nan` and `inf` are rejected. |
| Year | exactly four digits | `2O19` becomes `2019` |
| Name | must contain a letter | trims stray punctuation, collapses spaces |

`names_match(a, b)` compares two names with rapidfuzz `token_sort_ratio` (threshold 80). It
ignores case, punctuation and word order, so "Raza, Ali" matches "Ali Raza".

### 4.2 Extraction (`extract.py`)

For each field the extractor applies this logic, taken from the project brief.

| Field type | Condition | Status |
|---|---|---|
| checkbox | detector value `checked` if given, else ink at or above `checkbox_ink_min` | OK, with `checked` set |
| signature | ink at or above `signature_ink_min` | OK |
| signature | less ink (a stray dot is not a signature) | MISSING |
| text or date | no ink and no text | MISSING |
| text or date | ink present, OCR returned nothing | UNREADABLE |
| text or date | OCR confidence below 0.80 | UNREADABLE |
| text or date | placeholder such as `N/A`, `-`, `nil`, `xxxx` | INVALID |
| text or date | normalisation fails | INVALID |
| text or date | normalisation succeeds | OK, `value` set |

What `value` holds: the cleaned text for strings, an ISO date for dates, a stripped number for
numbers. When a value fails normalisation, text style fields keep the cleaned text as
`value` (so the rules can still see what was written), while dates and numbers get `None`.
`raw_text` is never modified: it stays exactly what OCR returned.

Defensive choices for real scans:

* If OCR returned text but `ink_ratio` is near zero, the text wins and a note is added.
  A default `ink_ratio` of 0.0 must never wipe out a successful read.
* One bad field never stops the whole form. Errors are caught per field and the field is
  marked UNREADABLE.

**Thresholds.** Defaults live in `ExtractionConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `ocr_conf_min` | 0.80 | below this, a field with ink is UNREADABLE (from the brief) |
| `repair_below_conf` | 0.90 | OCR confusion repair is attempted only below this |
| `ink_empty_below` | 0.002 | text and date boxes with less ink are empty |
| `signature_ink_min` | 0.01 | signature boxes need at least this much ink |
| `checkbox_ink_min` | 0.08 | used only if the detector did not set `checked` |

The three ink numbers are starting points, not facts. Run the calibration tool on your
labels and copy its suggestions:

```
python -m formsense.nlp.calibrate data/synthetic/dev/labels
```

It prints the ink distribution for empty versus filled boxes and suggests a threshold
between them. If the two groups overlap it prints `None` and you should inspect by hand.
Run it again on the real scanned forms, because real ink levels will differ.

### 4.3 Label mapper (`field_mapper.py`)

After template alignment, field boxes are known by position, so most fields need no label
mapping. The mapper is the safety net for when position is not enough: a different template
version prints different wording, or alignment was poor.

Three stages, cheapest first, each running only if the previous found nothing:

1. **Exact.** The normalised label equals the field's template label, its name written as
   words, or a listed alias. Normalisation lowercases and drops dots and apostrophes, so
   `D.O.B` becomes `dob` and `Father's Name:` becomes `fathers name`.
2. **Fuzzy.** rapidfuzz against all those texts, with look alike characters folded
   (`l`, `1` and `i` are treated alike) so `CNlC` still finds `cnic`. Needs a score of at
   least 0.86.
3. **Embedding.** MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) cosine similarity, for
   different wording such as "Date you were born". Needs a similarity of at least 0.60.
   The model is loaded lazily. If the package is not installed or the model cannot be
   downloaded, the mapper logs one warning and carries on with stages 1 and 2.

The mapper never guesses. If the best match is weak, or too close to the runner up (margin
0.04), it returns `field=None` with method `ambiguous` or `none` so the caller can send the
label to review. `map_labels_one_to_one` assigns every label on a page to a different field,
and gives the field to the more specific label on ties.

To add wording, edit `nlp/label_aliases.yaml`. Keep it in sync with the label variants the
generator can print (`--label-variants`).

## 5. Week 4 in detail: `rules/`

### 5.1 Rules are data (`rules.yaml`)

Every rule is a block in `rules.yaml`:

```yaml
- id: underage
  check: min_age
  type: inconsistency
  severity: error
  message: "The date of birth makes the applicant younger than {min_age}."
  description: Applicant must be at least the minimum age on the day of the audit.
  params:
    dob: date_of_birth
    min_age: 16
```

To change the minimum age, edit `min_age`. To rename a field, edit the field name. To
reword a message, edit `message`. No Python changes needed. When the engine starts it
validates the whole file and refuses to run with a clear error if a rule has a duplicate id,
an unknown check, a missing key or a bad severity. A test also confirms that every id, type
and severity in the YAML matches `codes.py`, so the two cannot drift apart.

Messages contain field labels, never field values. A CNIC or phone number can never leak
into a message, a log line or the LLM prompt through the rules. A test checks this.

### 5.2 How a run works (`engine.py`)

1. **Field level rules**, per field in template order. Rules run in the order they appear
   in the YAML. The first blocking hit wins: the field gets one issue and is marked "bad".
   Bad fields are hidden from the later cross field checks.
2. **Selection rules** for checkbox groups (nothing ticked, several ticked).
3. **Cross field rules.** Each one silently skips any field that is missing, unreadable or
   already bad.

This is the cascade control. If the CNIC box is empty, the form gets one message,
"CNIC is required but has been left empty", and not an extra confusing
"CNIC does not match gender". Only the real, independent problem is reported.

A rule marked `blocking: false` (currently `correction_mark`) is reported without hiding the
field, because a struck out and rewritten value can still be a valid value.

Robustness:

* Each rule runs in isolation. If one raises an error it is logged (rule id only) and the
  rest still run. Tests use `raise_errors=True` so bugs are visible there.
* Fields missing from the input are treated as MISSING with a warning. `strict=True` raises
  instead.
* Fields in the input that are not in the template are ignored.
* At start up a lint step warns about rules that refer to fields or groups the template does
  not have, because such a rule can never fire.
* The engine reads `FieldResult.value` when set and falls back to `raw_text`.
* "Today" is a parameter, so tests and evaluations are repeatable.

### 5.3 The 19 rules

| Rule id | Type | Severity | What it checks |
|---|---|---|---|
| `missing_required` | missing | error | Required text, date or signature field is empty. |
| `unreadable_field` | unreadable | warning | Ink present but OCR confidence too low. |
| `placeholder_value` | format | error | `N/A`, `nil`, `none`, dashes, a run of `x`. |
| `invalid_cnic` | format | error | Not `#####-#######-#`. |
| `invalid_email` | format | error | Not a valid email shape. |
| `invalid_phone` | format | error | Not `03XX-XXXXXXX`. |
| `invalid_date` | format | error | Date does not parse or does not exist. |
| `invalid_number` | format | error | Marks or percentage not numeric, or year not four digits. |
| `correction_mark` | format | warning | Struck out and rewritten value. Needs a vision flag (see section 11). |
| `no_selection` | selection | error | Nothing ticked in gender or program. |
| `multiple_selections` | selection | error | More than one ticked in a group. |
| `underage` | inconsistency | error | Applicant younger than 16 on the audit date. |
| `declaration_date_future` | inconsistency | error | Declaration dated after today. |
| `declaration_before_dob` | inconsistency | error | Declaration earlier than date of birth. |
| `cnic_gender_parity` | inconsistency | error | CNIC last digit odd means male, even means female. |
| `year_order` | inconsistency | error | Matric year must be earlier than intermediate year. |
| `marks_exceed_total` | inconsistency | error | Obtained marks higher than total. |
| `percentage_mismatch` | inconsistency | error | Stated percentage differs from obtained divided by total by more than 1 point. |
| `name_mismatch` | inconsistency | error | Declaration name does not fuzzy match the applicant name. |

Print the live catalogue at any time with `python -m formsense.rules.engine --list`.

### 5.4 Adding a rule

Most new rules need only a YAML block that reuses an existing `check`. For example, a second
CNIC style check is another `pattern` rule. A new kind of check is one short function in
`engine.py` registered with `@field_check("name")`, `@group_check("name")` or
`@cross_check("name")`, then used from the YAML. If the new rule is a real rule id, add it to
`codes.py` as well; the parity test will tell you if you forget.

## 6. Decisions made for real world use

| Decision | Why |
|---|---|
| Template driven, no hardcoded field lists in `nlp/` or in the engine core | A second form family needs a new template and a new `rules.yaml`, not new code. |
| Strict normalisation, repair gated by OCR confidence | Repair that always runs would hide genuine applicant mistakes. |
| Rules as YAML with start up validation | Non programmers can change thresholds; a typo fails loudly at start up instead of silently skipping a rule. |
| Cascade control through "bad" fields | One empty box produces one clear message, not five. |
| Every rule and every field isolated from failures | A single odd input cannot crash a whole audit. |
| Messages and logs carry labels, never values | Real forms contain personal data. |
| `today` is an argument | Age and future date checks stay testable and repeatable. |
| Ink thresholds are calibrated from data, not guessed forever | Real scans will have different ink levels from synthetic ones. |
| Embedding model is optional and lazy | The system still works offline and on a laptop without the model. |
| `raw_text` is never modified | The report can always show exactly what was read. |

## 7. Differences from `reference_validator.py`

The engine reproduces the reference validator exactly on every case the generator can make
(see the tests below). It differs deliberately in these edge cases, all of which only matter
for real input:

* An optional field that is empty is not flagged as a format error. The reference would
  report an invalid CNIC for an empty optional text field.
* A required field with status OK but no text is reported as `missing_required`.
* Numbers such as `-5`, `nan`, `inf` and `1e3` are rejected. The reference accepts them
  because Python's `float()` does.
* Thousands commas (`1,050`) are accepted as numbers.
* The engine accepts more date formats than the reference (dotted dates, full month names,
  ISO), because the extractor stores ISO dates in `value`.
* The extractor also accepts `dateparser` styles such as `5th March 2005` and `2005/03/14`
  when they are unambiguous. The reference only knows three formats. If you want the stricter
  behaviour, set `use_dateparser=False` in `ExtractionConfig`.

## 8. Testing

Run everything from the `backend` folder:

```
python -m pytest tests -q
```

| Test file | What it proves |
|---|---|
| `test_normalize.py` | Every normaliser on valid, invalid and OCR noisy input. Strict mode does not hide mistakes. Repair never invents digits. |
| `test_extract.py` | The MISSING versus UNREADABLE rule from the brief, INVALID handling, repair gating by confidence, checkbox and signature logic, `raw_text` untouched, no exceptions on odd input. |
| `test_field_mapper.py` | Every template label maps to its own field, typos and aliases, no guessing, male is not confused with female, one to one assignment, graceful fallback when the model is missing. |
| `test_rule_engine.py` | YAML matches `codes.py`, one test per rule including boundary cases (turns 16 exactly today, declaration dated today), cascade control, privacy of messages, crash isolation, bad YAML errors, lint, and the comparisons below. |
| `test_audit.py` | End to end from OCR style readings to issues: clean forms pass, a blank signature is found from ink alone, unsure OCR is repaired while confident OCR is not. |

Three comparisons give confidence that the engine is right:

1. **Engine versus reference validator.** 1,600 randomly corrupted forms (0 to 4 errors, 19
   corruption types) are given to both. The sets of issues must be identical. During
   development this was also run on 30,000 forms with zero differences.
2. **Extraction does not change the answer.** The same corrupted forms are run directly and
   through the extractor; the issues must be identical.
3. **Engine versus the generator's recorded issues.** If `data/synthetic/dev/labels` and
   `data/synthetic/test/labels` exist, the engine is run on every label and must reproduce
   the recorded issues. `correction_mark` is excluded because it needs a vision signal.

**Reference date for test 3.** The "underage" and "future date" rules depend on today's
date. The test looks for the generator's reference date in a constant such as `TODAY` in the
generator modules. If it cannot find one it prints a warning and skips those two rules. To
compare them too, set the date the generator used:

```
$env:FORMSENSE_TODAY = "2026-11-06"     # PowerShell, use the generator's date
python -m pytest tests/test_rule_engine.py -q -k recorded
```

Tests use the real template at `data/templates/admission_form.json` when it exists and fall
back to the bundled `tests/fixtures/mini_template.json` otherwise.

## 9. Commands

```
# install what is new
pip install pyyaml rapidfuzz dateparser numpy pytest
pip install sentence-transformers          # optional: enables the embedding stage

# audit one label file and compare with the recorded issues
python -m formsense.rules.engine data/synthetic/dev/labels/dev_0007.json --today 2026-11-06

# print the rule catalogue
python -m formsense.rules.engine --list

# audit raw field readings (runs extraction first)
python -m formsense.rules.audit readings.json --today 2026-11-06

# suggest ink thresholds from labelled forms
python -m formsense.nlp.calibrate data/synthetic/dev/labels
```

## 10. Contract with teammates

**For Ayesha (vision).** Please deliver one `FieldResult` per template field with
`raw_text`, `ocr_confidence` (0 to 1, `None` for checkbox and signature), `ink_ratio`, and
`checked` for checkboxes. Leave `status` and `value` at their defaults; Weeks 3 and 4 fill
them. If a field cannot be cropped at all, omit it: it will be treated as MISSING and logged.
To enable the `correction_mark` rule, pass `flags={"field_name": {"struck_through"}}` to
`audit()` when the vision side detects a crossed out value.

**For the LLM explainer (Week 5).** Use `result.issues`. Each issue has an id, a type, a
severity, field names, a safe message with no personal data, and evidence boxes. Values are
available in `result.fields` and must be masked before any LLM call.

**For the report and evaluation (Weeks 6 and 7).** `audit()` returns everything needed for
the checklist and for precision, recall and F1: compare `issue_key(issue)` (rule id plus the
set of fields) against the generator's labels, as the tests already do.

## 11. Known limitations and open questions

* **`correction_mark` is dormant.** The generator records it, but nothing in the text or
  ink numbers can detect a strike through. It fires only when the vision side passes the
  `struck_through` flag. Decide with Ayesha whether to build that detector or to report it as
  a limitation.
* **Ink thresholds are uncalibrated defaults** until you run the calibration tool on your
  data and on the real scans.
* **Label aliases** must be kept in sync with the generator's label variants.
* **The three extra matric fields** (`matric_total`, `matric_obtained`, `matric_percentage`)
  still need sign off from the team, because the rules and the extractor now depend on them.
* **Gaps already listed for the generator** (Urdu text, overflowing text, upside down pages,
  duplicate values across fields) are also not handled by the rules.
* **Single form family.** The rules file describes the admission form. A different form
  needs its own template and rules file.
* **Name matching is fuzzy.** Two different people with similar names would pass; two
  spellings of the same name with large differences would fail. The threshold is in the YAML.
* **Embedding stage untested against the real model here.** The tests use a stand in
  embedder. Run the mapper once on a machine with `sentence-transformers` installed and
  check a few labels.

## 12. What comes next

Week 5 is the LLM explainer and PII masking in `llm/`. It will take `AuditResult.issues`
and fields, mask CNIC, phone and email, and ask the model for a plain English explanation.
The issue messages and the evidence boxes produced here are designed to be passed to it
as they are.
