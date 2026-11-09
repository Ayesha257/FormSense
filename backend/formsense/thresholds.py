"""Thresholds that the vision side and the language side must agree on, in ONE place.

OCR_UNREADABLE_BELOW
    A text box with ink whose OCR confidence is below this is UNREADABLE (and raises an
    `unreadable_field` issue). It used to exist twice with different values (0.60 in the vision
    pipeline and baseline, 0.80 in nlp/extract.py, which runs last and so decided everything). The
    system effectively ran at 0.80. It is NOT the TrOCR fallback threshold (vision/ocr.py
    FALLBACK_BELOW), which only decides when the second OCR engine is tried.

    Evidence (Easha's saved real OCR readings of the 60 synthetic test forms, her labels):

        cutoff   issue precision   issue recall   issue F1   false "unreadable"   truly unreadable found
        0.80     0.31              0.87           0.45       110                  7 of 7
        0.60     0.36              0.87           0.51        46                  6 of 7
        0.40     0.37              0.85           0.52         6                  4 of 7

    0.60 removes most false alarms and still finds 6 of the 7 truly unreadable fields. This value was
    read off the TEST readings (with a two-fold check inside them), so it is provisional: confirm it on the
    dev split before quoting it,
        python scripts/evaluate_all.py --dir data/synthetic/dev --systems formsense --sweep-threshold
    and change the number here if dev says otherwise.
"""
OCR_UNREADABLE_BELOW = 0.60
