"""Baseline: plain full-page OCR + keyword/regex matching. No layout, no vision.

This is what "just run OCR on the form" gives you, and it is what FormSense
is compared against:

  1. OCR the whole page (same PaddleOCR + same cleanup as FormSense).
  2. Join the OCR boxes into text lines, top to bottom.
  3. For each field, find its printed label in the text and take whatever
     follows it on that line (or the next line) as the value.
  4. Regex fallback: a field whose label wasn't found grabs the first unused
     match of its pattern (CNIC, date, email, ...) anywhere in the text.

What it cannot do, by design (these are the multimodal parts):
  - know which box a value was written in, so values can land in the wrong field
  - see tick marks (it only guesses from characters like "x" or "v" before
    an option) or signatures (it assumes every form is signed)
  - tell an empty field from a field it failed to find (both become MISSING)

Output uses the same FieldResult schema as the vision pipeline, so the rules,
report and evaluation code treat both systems identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from formsense.baseline import patterns as P
from formsense.ingest.loader import preprocess
from formsense.schema import FieldResult, FieldSpec, FieldStatus, FieldType, FormTemplate
from formsense.vision.ocr import OCRBox, PaddleReader

UNREADABLE_BELOW = 0.60   # same threshold as the vision pipeline
TRIM = " :.-_|;,"


@dataclass
class TextRow:
    """One line of page text, built from OCR boxes at the same height."""
    boxes: list[OCRBox]
    text: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)  # char span of each box

    def __post_init__(self):
        self.boxes.sort(key=lambda b: b.x0)
        parts, pos = [], 0
        for b in self.boxes:
            t = b.text.strip()
            self.spans.append((pos, pos + len(t)))
            parts.append(t)
            pos += len(t) + 1
        self.text = " ".join(parts)

    def boxes_in(self, start: int, end: int) -> list[OCRBox]:
        return [b for b, (s, e) in zip(self.boxes, self.spans) if s < end and e > start]


@dataclass
class BaselineOutput:
    fields: list[FieldResult]
    rows: list[TextRow]

    @property
    def text(self) -> str:
        return "\n".join(r.text for r in self.rows)


def group_rows(boxes: list[OCRBox]) -> list[TextRow]:
    """Plain-text reading order: boxes whose vertical centres line up form a row."""
    boxes = sorted((b for b in boxes if b.text.strip()), key=lambda b: b.cy)
    rows: list[list[OCRBox]] = []
    for b in boxes:
        if rows:
            last = rows[-1]
            cy = float(np.mean([x.cy for x in last]))
            h = max(max(x.height for x in last), b.height)
            if abs(b.cy - cy) <= 0.5 * h:
                last.append(b)
                continue
        rows.append([b])
    return [TextRow(r) for r in rows]


def _union(boxes: list[OCRBox]) -> tuple[int, int, int, int]:
    if not boxes:
        return (0, 0, 0, 0)
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


class BaselineAuditor:
    def __init__(self, template: FormTemplate, reader: PaddleReader | None = None):
        self.template = template
        self._reader = reader
        self.labels = {f.name: P.label_regex(f.label) for f in template.fields}

    @property
    def reader(self) -> PaddleReader:
        if self._reader is None:
            self._reader = PaddleReader()
        return self._reader

    # ---------------- public ----------------

    def run(self, page: np.ndarray, clean: bool = True, max_side: int = 1600) -> BaselineOutput:
        gray = preprocess(page) if clean else page
        # full-page OCR is slow on big photos; shrinking to 1600 px reads the same text ~2.5x faster
        s = min(1.0, max_side / max(gray.shape[:2]))
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        boxes = self.reader.boxes(small)
        for b in boxes:  # back to the page's own pixel coordinates
            b.x0, b.y0, b.x1, b.y1 = b.x0 / s, b.y0 / s, b.x1 / s, b.y1 / s
        rows = group_rows(boxes)
        return BaselineOutput(self.extract(rows), rows)

    def extract(self, rows: list[TextRow]) -> list[FieldResult]:
        """Text rows -> FieldResults (no image needed; also used by tests)."""
        hits = self._find_labels(rows)
        used: set[tuple[int, int, int]] = set()     # (row, start, end) already taken
        out: dict[str, FieldResult] = {}

        # longest labels first so "Date of Birth" claims its line before "Date"
        for spec in sorted(self.template.fields, key=lambda f: -len(f.label)):
            if spec.type == FieldType.CHECKBOX:
                out[spec.name] = self._checkbox(spec, rows, hits)
            elif spec.type == FieldType.SIGNATURE:
                # plain OCR text has no way to see a signature: assume it is there
                out[spec.name] = FieldResult(name=spec.name, bbox=(0, 0, 0, 0),
                                             value="signed", status=FieldStatus.OK)
            else:
                out[spec.name] = self._text_field(spec, rows, hits, used)

        self._regex_fallback(rows, out, used)
        return [out[f.name] for f in self.template.fields]

    # ---------------- steps ----------------

    def _find_labels(self, rows):
        """Every place any field label appears: {field: [(row, start, end)]}."""
        hits: dict[str, list[tuple[int, int, int]]] = {}
        for name, rx in self.labels.items():
            if rx is None:
                continue
            hits[name] = [(i, m.start(), m.end())
                          for i, r in enumerate(rows) for m in rx.finditer(r.text)]
        return hits

    def _claimed(self, hits, exclude: str) -> list[tuple[int, int, int]]:
        return [h for name, hs in hits.items() if name != exclude for h in hs]

    def _text_field(self, spec: FieldSpec, rows, hits, used) -> FieldResult:
        result = FieldResult(name=spec.name, bbox=(0, 0, 0, 0), status=FieldStatus.MISSING)
        others = self._claimed(hits, spec.name)
        for (ri, s, e) in hits.get(spec.name, []):
            # skip a match that sits inside a longer label already claimed
            if any(ri == r and s >= os_ and e <= oe and (oe - os_) > (e - s)
                   for (r, os_, oe) in others if (r, os_, oe) in used):
                continue
            used.add((ri, s, e))
            row = rows[ri]
            # value = text after the label, up to the next label on the same line
            end = min([os_ for (r, os_, _) in others if r == ri and os_ >= e] + [len(row.text)])
            value_start, value_end = e, end
            value = row.text[e:end].strip(TRIM)
            value_row = ri
            if not value and ri + 1 < len(rows) and not any(r == ri + 1 for (r, _, _) in others):
                # nothing after the label: try the next line (label-above-value layout)
                nxt = rows[ri + 1]
                if not any(r == ri + 1 for (r, _, _) in hits.get(spec.name, [])):
                    value_row, value_start, value_end = ri + 1, 0, len(nxt.text)
                    value = nxt.text.strip(TRIM)
            if value:
                pat = P.pattern_for(spec)
                m = pat.search(value) if pat else None
                if m:
                    off = rows[value_row].text.find(value, value_start)
                    value_start, value_end = off + m.start(), off + m.end()
                    value = m.group(0)
                self._fill(result, rows[value_row], value_row, value_start, value_end, value, used)
            return result
        return result

    def _fill(self, result, row: TextRow, ri: int, start: int, end: int, value: str, used):
        boxes = row.boxes_in(start, end)
        conf = float(np.mean([b.score for b in boxes])) if boxes else 0.0
        result.raw_text = value
        result.ocr_confidence = round(conf, 3)
        result.bbox = _union(boxes)
        result.status = FieldStatus.UNREADABLE if conf < UNREADABLE_BELOW else FieldStatus.OK
        used.add((ri, start, end))

    def _checkbox(self, spec: FieldSpec, rows, hits) -> FieldResult:
        result = FieldResult(name=spec.name, bbox=(0, 0, 0, 0))
        found = hits.get(spec.name, [])
        if not found:
            result.checked, result.value = False, "unchecked"   # can't see it: say no
            return result
        ri, s, _ = found[0]
        before = rows[ri].text[:s]
        result.checked = bool(P.TICK.search(before))
        result.value = "checked" if result.checked else "unchecked"
        result.bbox = _union(rows[ri].boxes_in(s, s + 1))
        return result

    def _regex_fallback(self, rows, out: dict[str, FieldResult], used):
        """Fields still empty grab the first unused pattern match on the page."""
        for spec in self.template.fields:
            r = out[spec.name]
            if spec.type in (FieldType.CHECKBOX, FieldType.SIGNATURE) or r.raw_text:
                continue
            pat = P.pattern_for(spec)
            if pat is None or pat in (P.NUMBER, P.YEAR):   # too generic to guess
                continue
            for ri, row in enumerate(rows):
                hit = next((m for m in pat.finditer(row.text)
                            if not any(u[0] == ri and m.start() < u[2] and m.end() > u[1]
                                       for u in used)), None)
                if hit:
                    self._fill(r, row, ri, hit.start(), hit.end(), hit.group(0), used)
                    break
