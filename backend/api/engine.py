"""Runs the existing FormSense backend for the web API.

One worker thread does all model work (OCR, rules, retrieval, LLM), one form at a time:
PaddleOCR and the embedding model are not thread-safe, and a laptop CPU gains nothing from
running two forms at once. The HTTP layer queues jobs here and polls their status.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from api import config
from api.store import HistoryStore

log = logging.getLogger("formsense.api")

# Plain-language names for the rule ids in formsense/rules/codes.py
RULE_TITLES = {
    "missing_required": "Required item left empty",
    "unreadable_field": "Handwriting is hard to read",
    "placeholder_value": "Placeholder instead of an answer",
    "invalid_cnic": "CNIC number is not valid",
    "invalid_email": "Email address is not valid",
    "invalid_phone": "Mobile number is not valid",
    "invalid_date": "Date is not valid",
    "invalid_number": "Number is not valid",
    "correction_mark": "Value was corrected by hand",
    "no_selection": "Nothing ticked",
    "multiple_selections": "More than one box ticked",
    "underage": "Applicant is under 16",
    "declaration_date_future": "Declaration date is in the future",
    "declaration_before_dob": "Declaration date is before the date of birth",
    "cnic_gender_parity": "CNIC does not match the gender",
    "year_order": "Exam years are in the wrong order",
    "marks_exceed_total": "Obtained marks exceed the total",
    "percentage_mismatch": "Percentage does not match the marks",
    "name_mismatch": "Declaration name does not match",
}
SECTION_TITLES = {"personal": "Personal details", "program": "Program applied for",
                  "academic": "Academic record", "declaration": "Declaration"}
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
STAGES = ("queued", "reading", "analyzing", "explaining", "done")


def _dump(m) -> Any:
    return m.model_dump(mode="json")


class Engine:
    def __init__(self, store: HistoryStore):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="formsense")
        self.state = "idle"            # idle -> loading -> ready | error
        self.state_error: Optional[str] = None
        self._fs: dict[bool, Any] = {}
        self._ocr: dict[bool, Any] = {}
        self.trocr_error: Optional[str] = None      # set when the handwriting model can't be loaded
        self._retriever = None
        self._template = None
        self._spec_json: dict[str, dict] = {}
        self._labels: dict[str, str] = {}
        self._queue = 0
        self._qlock = threading.Lock()

    # ------------------------------------------------------------------ models
    def template(self):
        if self._template is None:
            from formsense.rag.retriever import field_labels
            from formsense.vision.template import load_template
            self._template, self._blank = load_template(config.TEMPLATE)
            raw = json.loads(Path(config.TEMPLATE).read_text(encoding="utf-8"))
            self._spec_json = {f["name"]: f for f in raw["fields"]}
            self._groups = raw.get("groups") or {}
            self._labels = field_labels(config.TEMPLATE)
        return self._template

    def blank(self):
        self.template()
        return self._blank

    def spec_json(self) -> dict[str, dict]:
        self.template()
        return self._spec_json

    def retriever(self):
        if self._retriever is None:
            from formsense.rag import GuidelineRetriever
            self._retriever = GuidelineRetriever().ensure()
        return self._retriever

    def ocr(self, use_trocr: bool):
        if use_trocr not in self._ocr:
            ocr = _safe_field_ocr(use_trocr, self)
            other = self._ocr.get(not use_trocr)
            if other is not None and other._paddle is not None:
                ocr._paddle = other._paddle          # one PaddleOCR model in memory, not two
            self._ocr[use_trocr] = ocr
        return self._ocr[use_trocr]

    def formsense(self, use_trocr: bool):
        if use_trocr not in self._fs:
            from formsense.pipeline import FormSense
            self._fs[use_trocr] = FormSense(config.TEMPLATE, use_trocr=use_trocr,
                                            ocr=self.ocr(use_trocr), retriever=self.retriever())
        return self._fs[use_trocr]

    def warm_up(self) -> Future:
        """Load the template, PaddleOCR and the guidelines index in the background."""
        def _load():
            self.state = "loading"
            try:
                self.template()
                self.ocr(True).paddle               # PaddleOCR models (TrOCR loads on first use)
                self.retriever()
                self.state = "ready"
            except Exception as e:                  # the app still works; the first job will retry
                self.state, self.state_error = "error", str(e)
                log.exception("warm-up failed")
        return self.pool.submit(_load)

    # ------------------------------------------------------------------ jobs
    @property
    def queued(self) -> int:
        return self._queue

    def submit(self, form_id: str) -> None:
        with self._qlock:
            self._queue += 1
        self.pool.submit(self._run_job, form_id)

    def run_sync(self, fn, *args, **kwargs):
        """Run model work on the worker thread and wait for it (keeps models single-threaded)."""
        return self.pool.submit(fn, *args, **kwargs).result()

    def _stage(self, form_id: str, stage: str, progress: float):
        self.store.update_meta(form_id, status="processing", stage=stage, progress=progress)

    def _run_job(self, form_id: str) -> None:
        try:
            self._process(form_id)
        except Exception as e:
            log.error("form %s failed: %s\n%s", form_id, e, traceback.format_exc())
            self.store.update_meta(form_id, status="failed", stage="failed", error=friendly_error(e))
        finally:
            with self._qlock:
                self._queue = max(0, self._queue - 1)

    def _process(self, form_id: str) -> None:
        from formsense.ingest.loader import load_bytes
        meta = self.store.meta(form_id)
        opts = meta["options"]
        t0 = time.time()

        self._stage(form_id, "reading", 0.12)
        data, filename = self.store.original_bytes(form_id)
        pages = load_bytes(data, filename)
        if not pages:
            raise ValueError("The file has no pages.")
        page_idx = min(max(int(opts.get("page", 1)) - 1, 0), len(pages) - 1)
        page = pages[page_idx]
        cv2.imwrite(str(self.store.path(form_id, "page.png")), page)
        self._thumb(page, self.store.path(form_id, "thumb.jpg"))
        self.store.update_meta(form_id, pages=len(pages), page=page_idx + 1)

        self._stage(form_id, "analyzing", 0.3)
        self.template()
        fs = self.formsense(bool(opts.get("use_trocr", True)))
        today = parse_date(opts.get("today")) or dt.date.today()
        result = fs.run(page, today=today, form_id=Path(filename).stem)
        if opts.get("use_trocr", True) and self.trocr_error:
            self.store.update_meta(form_id, handwriting_reader="unavailable")
        cv2.imwrite(str(self.store.path(form_id, "aligned.png")), result.vision.aligned)
        self.store.write_json(form_id, "vision.json", [_dump(f) for f in result.vision.fields])
        alignment = {"ok": bool(result.vision.alignment.ok), "inliers": int(result.vision.alignment.inliers),
                     "matches": int(result.vision.alignment.matches)}
        self._save_audit(form_id, result.fields, result.issues, result.citations, result.notes, alignment,
                         corrections={}, seconds=round(time.time() - t0, 1))

        if opts.get("explain", True):
            self._stage(form_id, "explaining", 0.85)
            self.explain(form_id, use_rag=bool(opts.get("use_rag", True)))

        self.store.update_meta(form_id, status="done", stage="done", progress=1.0,
                               seconds=round(time.time() - t0, 1))

    @staticmethod
    def _thumb(img: np.ndarray, path: Path, width: int = 360) -> None:
        h, w = img.shape[:2]
        small = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 82])

    # ------------------------------------------------------------------ audit results
    def _save_audit(self, form_id, fields, issues, citations, notes, alignment, corrections, seconds=None):
        audit = {"fields": [_dump(f) for f in fields], "issues": [_dump(i) for i in issues],
                 "citations": {k: _dump(c) for k, c in citations.items()}, "notes": notes,
                 "alignment": alignment, "corrections": corrections}
        prev = self.store.read_json(form_id, "audit.json") or {}
        if seconds is None:
            seconds = prev.get("seconds")
        audit["seconds"] = seconds
        if prev.get("explanation"):
            audit["explanation"] = prev["explanation"]
            audit["explanation_stale"] = bool(corrections != prev.get("corrections", {})) or prev.get("explanation_stale", False)
        self.store.write_json(form_id, "audit.json", audit)
        payload = self.payload(form_id, audit)
        self.store.update_meta(form_id, summary={
            "verdict": payload["verdict"], "issues": payload["counts"]["issues"],
            "fields": payload["counts"]["fields"], "seconds": seconds,
            "top_issues": [i["title"] for i in payload["issues"][:3]],
            "issue_ids": [i["id"] for i in payload["issues"]]})

    def payload(self, form_id: str, audit: Optional[dict] = None) -> dict:
        """Everything the result page needs, built from audit.json."""
        from formsense.pipeline import issue_key
        from formsense.schema import FieldResult, Issue
        from formsense.summary import checklist, field_counts, issue_counts, plain_summary
        template = self.template()
        audit = audit or self.store.read_json(form_id, "audit.json")
        meta = self.store.meta(form_id)
        fields = [FieldResult(**f) for f in audit["fields"]]
        issues = [Issue(**i) for i in audit["issues"]]
        by_name = {f.name: f for f in fields}
        corrections = audit.get("corrections") or {}

        field_rows = []
        for spec in template.fields:
            r = by_name.get(spec.name)
            if r is None:
                continue
            raw = self._spec_json.get(spec.name, {})
            field_rows.append({
                "name": spec.name, "label": spec.label, "type": spec.type.value,
                "kind": raw.get("kind", spec.type.value), "section": spec.section,
                "section_title": SECTION_TITLES.get(spec.section or "", (spec.section or "Other").title()),
                "group": spec.group, "group_label": (self._groups.get(spec.group) or {}).get("label") if spec.group else None,
                "required": spec.required, "hint": raw.get("hint"),
                "status": r.status.value, "raw_text": r.raw_text, "value": r.value,
                "confidence": r.ocr_confidence, "ink": r.ink_ratio, "checked": r.checked,
                "bbox": list(r.bbox), "notes": (audit.get("notes") or {}).get(spec.name, []),
                "corrected": spec.name in corrections})

        # the explainer answers one entry per issue, in the order the issues were given
        exp = audit.get("explanation") or {}
        exp_items = exp.get("issues") or []
        per_issue = {}
        for idx, i in enumerate(issues):
            if idx < len(exp_items) and exp_items[idx].get("issue_id") == i.id:
                per_issue[idx] = exp_items[idx].get("explanation")
        order = sorted(range(len(issues)), key=lambda k: SEVERITY_ORDER.get(issues[k].severity.value, 9))
        issue_rows = []
        for n, idx in enumerate(order, 1):
            i = issues[idx]
            cite = (audit.get("citations") or {}).get(issue_key(i))
            evidence = [list(b) for b in i.evidence if b[2] > 0 and b[3] > 0] or \
                       [list(by_name[f].bbox) for f in i.fields if f in by_name]
            issue_rows.append({
                "n": n, "key": issue_key(i), "id": i.id, "title": RULE_TITLES.get(i.id, i.id.replace("_", " ").capitalize()),
                "type": i.type.value, "severity": i.severity.value, "fields": i.fields,
                "field_labels": list(OrderedDict.fromkeys(self._labels.get(f, f) for f in i.fields)),
                "message": i.message, "evidence": evidence, "explanation": per_issue.get(idx),
                "citation": {"source": cite["source"], "text": cite["text"]} if cite else None})

        counts_i = issue_counts(issues)
        counts_f = field_counts(fields, template)
        alignment = audit.get("alignment") or {"ok": True}
        if not alignment.get("ok"):
            verdict = "unaligned"
        elif counts_i["error"]:
            verdict = "fail"
        elif counts_i["warning"] or counts_i["info"]:
            verdict = "review"
        else:
            verdict = "pass"

        out = {
            "id": form_id, "filename": meta["filename"], "created_at": meta["created_at"],
            "updated_at": meta["updated_at"], "status": meta["status"], "stage": meta["stage"],
            "options": meta["options"], "pages": meta.get("pages", 1), "page": meta.get("page", 1),
            "seconds": audit.get("seconds"), "template": {"width": template.width, "height": template.height},
            "alignment": alignment, "verdict": verdict,
            "counts": {"issues": counts_i, "fields": counts_f},
            "fields": field_rows, "issues": issue_rows,
            "checklist": checklist(fields, template, self._labels),
            "summary": plain_summary(fields, issues, template, self._labels),
            "explanation": audit.get("explanation"), "explanation_stale": audit.get("explanation_stale", False),
            "corrections": corrections, "sample": None,
        }
        if meta.get("sample"):
            out["sample"] = self._sample_truth(meta["sample"], issues)
        return out

    def _sample_truth(self, sample: str, issues) -> Optional[dict]:
        from formsense.scoring import score_issues
        label = sample_label(sample)
        if not label:
            return None
        s = score_issues(issues, label["issues"])
        return {"name": sample, "profile": (label.get("capture") or {}).get("profile"),
                "true_issues": len(label["issues"]), "found": s["tp"], "false_alarms": s["fp"],
                "missed": s["missed"], "extra": s["extra"]}

    # ------------------------------------------------------------------ corrections
    def apply_corrections(self, form_id: str, corrections: dict[str, dict]) -> dict:
        """Re-check the form with values typed by a person. Replaces any earlier corrections."""
        from formsense.rules.audit import audit as run_audit
        from formsense.schema import FieldResult, FieldStatus, FieldType
        template = self.template()
        specs = {f.name: f for f in template.fields}
        unknown = [n for n in corrections if n not in specs]
        if unknown:
            raise ValueError(f"Unknown field(s): {', '.join(unknown)}")
        from formsense.nlp.extract import ExtractionConfig
        cfg = ExtractionConfig()
        vision = [FieldResult(**f) for f in self.store.read_json(form_id, "vision.json")]
        fixed = []
        for r in vision:
            c = corrections.get(r.name)
            if c is None:
                fixed.append(r)
                continue
            spec = specs[r.name]
            r = r.model_copy()
            if spec.type == FieldType.CHECKBOX:
                r.checked = bool(c.get("checked"))
                r.value = "checked" if r.checked else "unchecked"
                r.status = FieldStatus.OK
            elif spec.type == FieldType.SIGNATURE:
                # the extractor decides "signed" from the amount of ink, so set that too
                signed = bool(c.get("signed"))
                r.status, r.value = (FieldStatus.OK, "signed") if signed else (FieldStatus.MISSING, None)
                r.ink_ratio = max(r.ink_ratio, 4 * cfg.signature_ink_min) if signed else 0.0
            else:
                text = (c.get("value") or "").strip()
                r.raw_text, r.value, r.ocr_confidence = text, None, 1.0     # a person typed it: fully confident
                r.status = FieldStatus.OK if text else FieldStatus.MISSING
                if not text:
                    r.ink_ratio = 0.0                                    # cleared on purpose: empty, not unreadable
            fixed.append(r)
        meta = self.store.meta(form_id)
        today = parse_date(meta["options"].get("today")) or dt.date.today()
        res = run_audit(fixed, template=config.TEMPLATE, today=today)
        citations = self._citations(res.issues)
        prev = self.store.read_json(form_id, "audit.json") or {}
        self._save_audit(form_id, res.fields, res.issues, citations, res.notes,
                         prev.get("alignment", {"ok": True}), corrections)
        return self.payload(form_id)

    def _citations(self, issues) -> dict:
        from formsense.pipeline import issue_key
        out = {}
        retriever = self.retriever()
        for i in issues:
            cites = retriever.for_issue(i, self._labels)
            if cites:
                out[issue_key(i)] = cites[0]
        return out

    # ------------------------------------------------------------------ explanation + report
    def explain(self, form_id: str, use_rag: bool = True) -> dict:
        from formsense.explain import LLMModelUnavailable, explain_report, is_offline
        report = self.report(form_id, with_explanation=False)
        error = None
        source = "template" if is_offline() else "llm"
        try:
            out = explain_report(report, config.TEMPLATE, use_rag=use_rag)
        except LLMModelUnavailable as e:
            error, source = str(e), "template"
            out = explain_report(report, config.TEMPLATE, use_rag=use_rag, offline=True)
        except Exception as e:                       # network down, quota, bad key: keep a usable text
            error, source = f"The AI explanation could not be generated ({type(e).__name__}: {e}).", "template"
            out = explain_report(report, config.TEMPLATE, use_rag=use_rag, offline=True)
        explanation = {**_dump(out), "source": source, "use_rag": use_rag, "error": error,
                       "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        audit = self.store.read_json(form_id, "audit.json")
        audit["explanation"], audit["explanation_stale"] = explanation, False
        self.store.write_json(form_id, "audit.json", audit)
        return explanation

    def report(self, form_id: str, with_explanation: bool = True):
        from formsense.llm.explanation_schema import ExplanationOutput
        from formsense.report.build import build_report
        from formsense.schema import FieldResult, GuidelineCitation, Issue
        audit = self.store.read_json(form_id, "audit.json")
        meta = self.store.meta(form_id)
        exp = audit.get("explanation") if with_explanation else None
        explanation = None
        if exp:
            explanation = ExplanationOutput(summary=exp["summary"], issues=exp.get("issues", []),
                                            overall_recommendation=exp.get("overall_recommendation"))
        return build_report(Path(meta["filename"]).stem,
                            [FieldResult(**f) for f in audit["fields"]],
                            [Issue(**i) for i in audit["issues"]], explanation,
                            [GuidelineCitation(**c) for c in (audit.get("citations") or {}).values()])

    def pdf(self, form_id: str) -> bytes:
        from formsense.explain import pdf_bytes
        return pdf_bytes(self.report(form_id), config.TEMPLATE)

    def checklist_csv(self, form_id: str) -> str:
        from formsense.summary import checklist_csv
        return checklist_csv(self.payload(form_id)["checklist"])

    # ------------------------------------------------------------------ guidelines
    def search_guidelines(self, q: str, k: int = 4) -> list[dict]:
        return [{"section": p.section, "source": p.source, "text": p.text, "score": round(float(p.score), 3)}
                for p in self.retriever().search(q, k=k)]

    def guideline_sections(self) -> list[dict]:
        from formsense.rag.retriever import load_chunks
        out: "OrderedDict[str, dict]" = OrderedDict()
        for d in load_chunks():
            sec = d.metadata.get("section") or d.metadata.get("source", "")
            entry = out.setdefault(sec, {"section": sec, "source": d.metadata.get("source", ""), "text": ""})
            entry["text"] = (entry["text"] + "\n\n" + d.page_content).strip()
        return list(out.values())


# ---------------------------------------------------------------------- helpers

def parse_date(value) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def sample_label(name: str) -> Optional[dict]:
    p = config.SAMPLE_DIR / "labels" / f"{Path(name).stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_samples(limit: int = config.MAX_SAMPLES) -> list[dict]:
    images = config.SAMPLE_DIR / "images"
    if not images.is_dir():
        return []
    out = []
    for p in sorted(images.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        label = sample_label(p.name) or {}
        issues = label.get("issues", [])
        out.append({"name": p.name, "form_id": label.get("form_id", p.stem),
                    "profile": (label.get("capture") or {}).get("profile"),
                    "reference_date": label.get("reference_date"),
                    "issues": len(issues),
                    "issue_titles": [RULE_TITLES.get(i["id"], i["id"]) for i in issues][:3]})
        if len(out) >= limit:
            break
    return out


def friendly_error(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    low = msg.lower()
    if "cannot identify" in low or "could not read" in low or "could not decode" in low or "failed to open" in low:
        return "This file could not be opened as an image or PDF. Upload a JPG, PNG or PDF of the form."
    if "unsupported file type" in low:
        return "This file type is not supported. Upload a JPG, PNG or PDF."
    return f"Checking failed: {msg}"


def _safe_field_ocr(use_trocr: bool, engine: "Engine"):
    """FieldOCR that keeps going with PaddleOCR alone when the TrOCR model can't be loaded
    (not downloaded yet, HF_HUB_OFFLINE=1, no internet, Hugging Face rate limit)."""
    from formsense.vision.ocr import FieldOCR

    class SafeFieldOCR(FieldOCR):
        def read(self, crop):
            try:
                return super().read(crop)
            except Exception as e:
                # Paddle read first and loaded fine; only the TrOCR load failed: drop TrOCR, keep Paddle's answer
                if self.use_trocr and self._trocr is None and self._paddle is not None:
                    self.use_trocr = False
                    engine.trocr_error = str(e).splitlines()[0][:300]
                    log.warning("handwriting reader (TrOCR) unavailable, reading with PaddleOCR only: %s",
                                engine.trocr_error)
                    return self.paddle.read(crop)
                raise

    return SafeFieldOCR(use_trocr=use_trocr)
