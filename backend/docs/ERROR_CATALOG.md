# Error catalog (what the generator injects and what the rule engine must catch)

`schema.py` is untouched. Each issue below is a rule id in `Issue.id` (lowercase of the name in the first column, `CNIC_GENDER_MISMATCH` is `cnic_gender_parity`),
and `Issue.type` is the coarse schema type: MISSING_REQUIRED = missing, UNREADABLE_FIELD = unreadable, all INVALID_* / PLACEHOLDER_VALUE / CORRECTION_MARK = format,
NO_SELECTION and MULTIPLE_SELECTIONS = selection, all cross field rules = inconsistency. Ids live in `formsense/rules/codes.py`.
`no_gender_or_program` now produces NO_SELECTION (not MISSING_REQUIRED). Fields: 20 from the shared template plus matric_total, matric_obtained, matric_percentage.

Every injector breaks a valid form in one way and records the expected Issue. Two injectors never touch the same field.
`rules/reference_validator.py` re-derives all issues from the ground truth; tests prove both agree on 300 forms.

## Real errors (ground truth Issues)
| IssueType | Severity | Injector (variants) |
|---|---|---|
| MISSING_REQUIRED | ERROR | missing_text (any of 17 text fields), missing_signature, signature_speck (stray dot only) |
| NO_SELECTION | ERROR | no_gender_or_program (no box ticked) |
| UNREADABLE_FIELD | WARNING | unreadable (smeared, scribbled, blurred ink) |
| PLACEHOLDER_VALUE | ERROR | placeholder (N/A, n/a, -, nil, XXXX, None, ......) in father name or address |
| INVALID_CNIC | ERROR | cnic_bad (1 to 3 digits dropped, extra digit, no dashes, spaces, O for 0 / l for 1 / S for 5 / B for 8) |
| INVALID_EMAIL | ERROR | email_bad (no @, no dot in domain, space, double @, no local part, trailing dot, double dot, comma) |
| INVALID_PHONE | ERROR | phone_bad (04XX prefix, too short, too long, no dash, landline, space instead of dash, email written in phone box) |
| INVALID_DATE | ERROR | date_bad (31/02, 30/02, 31/04, 29/02 non leap, day 45, month 13, 00/00/0000, month 00) in DOB or declaration date |
| INVALID_NUMBER | ERROR | number_bad (abc, 7OO, seven hundred, 9x0, 20l8, 1,0S0) in year, total, obtained, percentage |
| CORRECTION_MARK | WARNING | correction (value struck out and rewritten, image only) |
| UNDERAGE | ERROR | underage (age 8 to 15) |
| DECLARATION_DATE_FUTURE | ERROR | decl_future (1 to 400 days ahead) |
| DECLARATION_BEFORE_DOB | ERROR | decl_before_dob |
| CNIC_GENDER_MISMATCH | ERROR | cnic_gender (either the tick or the last CNIC digit is flipped) |
| MULTIPLE_SELECTIONS | ERROR | multiple (2 or 3 boxes ticked in gender or program) |
| YEAR_ORDER | ERROR | year_order (equal, swapped, matric later than inter) |
| MARKS_EXCEED_TOTAL | ERROR | marks_exceed (matric or inter, percentage consistent with the wrong marks) |
| PERCENTAGE_MISMATCH | ERROR | pct_mismatch (off by 3 to 20 points) |
| NAME_MISMATCH | ERROR | name_mismatch (different person, or same first name with other surname) |

Field status in ground truth: MISSING (no ink), UNREADABLE (ink, illegible), INVALID (readable but bad value), else OK.
Cross field issues leave the field status OK. If an input is MISSING / UNREADABLE / INVALID the dependent cross check is skipped.

## Benign variations (must NOT raise an issue, they measure false positives)
name_variant (upper or lower case, dropped middle name, one letter typo, trailing dot, reordered), mixed_date_style
(DD/MM/YYYY vs DD-MM-YYYY vs DD Mon YYYY), age_exactly_16, decl_today, full_marks (100.00), pct_symbol (`61.9%`),
faint_signature (light pencil, must count as present). Plus per form: 4 ink colours, 6 handwriting fonts, tick / cross / filled checkbox styles.

## Capture conditions (vision side robustness, recorded with the homography template to image)
scan_clean, scan_noisy (tilt, JPEG, uneven light), phone_good (perspective, shadow), phone_bad (strong tilt and perspective,
motion blur, low light, heavy JPEG). Optional `--label-variants` prints synonym labels (D.O.B, E-mail, Cell No.) to test label matching.

## Not injected yet (ideas to agree in Week 1 kickoff, tell me and I add them)
* Text outside its box, overlapping lines, two languages (Urdu) in one field
* Crossed out tick, tick between two boxes, form page missing or upside down
* Matric / inter year plausibility vs DOB, impossible marks totals (not 1200 / 1100)
* Duplicate values across fields (father name equals applicant name, phone equals CNIC)
* Real world: smudges, folds, staples, stamps over fields (needs the 15 real forms to calibrate)
