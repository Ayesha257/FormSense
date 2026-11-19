"""FastAPI app: REST endpoints under /api, and the built web app (frontend/dist) at /.

    GET    /api/health                       server, models and AI-explanation status
    GET    /api/template                     the admission form: fields, sections, size
    GET    /api/template/blank.png           blank form image
    GET    /api/stats                        dashboard numbers from the history
    GET    /api/samples                      generated sample forms (with known answers)
    GET    /api/samples/{name}/thumb.jpg     sample preview
    POST   /api/samples/{name}/check         check a sample form           -> job
    POST   /api/forms                        upload and check a form       -> job
    GET    /api/forms                        history
    GET    /api/forms/{id}                   status while processing, full result when done
    DELETE /api/forms/{id}                   remove from history
    GET    /api/forms/{id}/image/{kind}      page | aligned | thumb | original
    GET    /api/forms/{id}/fields/{name}/crop.png   close-up of one field
    PUT    /api/forms/{id}/corrections       re-check with values typed by a person
    POST   /api/forms/{id}/explain           (re)write the plain-English explanation
    GET    /api/forms/{id}/report.pdf|.json  audit report downloads
    GET    /api/forms/{id}/checklist.csv     checklist download
    GET    /api/guidelines                   all guideline sections
    GET    /api/guidelines/search?q=         guideline search (RAG retrieval)
"""
from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import __version__, config
from api.engine import RULE_TITLES, Engine, list_samples, parse_date, sample_label
from api.store import HistoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("formsense.api")

store = HistoryStore()
engine = Engine(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.mark_interrupted()
    if config.WARM_UP:
        engine.warm_up()
    yield
    engine.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="FormSense API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------- helpers

def _meta_or_404(form_id: str) -> dict:
    if not store.exists(form_id):
        raise HTTPException(404, "This form is not in the history. It may have been deleted.")
    return store.meta(form_id)


def _done_or_409(form_id: str) -> dict:
    meta = _meta_or_404(form_id)
    if meta["status"] != "done" or store.read_json(form_id, "audit.json") is None:
        raise HTTPException(409, "This form is still being checked.")
    return meta


def _png(img) -> Response:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Could not encode the image.")
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _job_options(use_trocr: bool, explain: bool, use_rag: bool, today: Optional[str], page: int) -> dict:
    d = parse_date(today)
    return {"use_trocr": use_trocr, "explain": explain, "use_rag": use_rag,
            "today": (d or dt.date.today()).isoformat(), "page": max(1, page)}


def _llm_status() -> dict:
    from formsense import groq_util
    from formsense.explain import has_llm_key
    key = has_llm_key()
    return {"available": key, "model": groq_util.text_model() if key else None}


# ---------------------------------------------------------------------- system

@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "models": engine.state, "models_error": engine.state_error,
            "handwriting_reader": {"available": engine.trocr_error is None, "error": engine.trocr_error},
            "queue": engine.queued, "llm": _llm_status(), "samples": len(list_samples(limit=10_000)),
            "max_upload_mb": config.MAX_UPLOAD_MB, "allowed": sorted(config.ALLOWED_EXT)}


@app.get("/api/template")
def template():
    t = engine.template()
    rows = engine.spec_json()
    return {"name": t.name, "width": t.width, "height": t.height,
            "fields": [{"name": f.name, "label": f.label, "type": f.type.value, "section": f.section,
                        "group": f.group, "required": f.required, "bbox": list(f.bbox),
                        "hint": rows.get(f.name, {}).get("hint")} for f in t.fields]}


@app.get("/api/template/blank.png")
def blank_png():
    return _png(engine.blank())


@app.get("/api/stats")
def stats():
    metas = store.list()
    done = [m for m in metas if m.get("status") == "done" and m.get("summary")]
    verdicts = Counter(m["summary"]["verdict"] for m in done)
    rules = Counter(i for m in done for i in m["summary"].get("issue_ids", []))
    secs = [m["summary"]["seconds"] for m in done if m["summary"].get("seconds")]
    today = dt.date.today()
    days = []
    for back in range(6, -1, -1):
        d = today - dt.timedelta(days=back)
        days.append({"date": d.isoformat(),
                     "count": sum(1 for m in metas if m.get("created_at", "")[:10] == d.isoformat())})
    return {"total": len(metas), "done": len(done),
            "failed": sum(1 for m in metas if m.get("status") == "failed"),
            "processing": sum(1 for m in metas if m.get("status") in ("queued", "processing")),
            "verdicts": {k: verdicts.get(k, 0) for k in ("pass", "review", "fail", "unaligned")},
            "issues_found": sum(sum(m["summary"]["issues"].values()) for m in done),
            "avg_seconds": round(sum(secs) / len(secs), 1) if secs else None,
            "top_rules": [{"id": r, "title": RULE_TITLES.get(r, r), "count": c} for r, c in rules.most_common(5)],
            "days": days}


# ---------------------------------------------------------------------- samples

@app.get("/api/samples")
def samples():
    return {"folder": str(config.SAMPLE_DIR), "samples": list_samples()}


def _sample_path(name: str) -> Path:
    p = config.SAMPLE_DIR / "images" / Path(name).name
    if not p.exists():
        raise HTTPException(404, "Sample not found. Generate samples first.")
    return p


@app.get("/api/samples/{name}/thumb.jpg")
def sample_thumb(name: str):
    p = _sample_path(name)
    cache = config.SAMPLE_DIR / ".thumbs" / f"{p.stem}.jpg"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        img = cv2.imread(str(p))
        if img is None:
            raise HTTPException(500, "Sample image could not be read.")
        engine._thumb(img, cache)
    return FileResponse(cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


class SampleOptions(BaseModel):
    use_trocr: bool = True
    explain: bool = True
    use_rag: bool = True


@app.post("/api/samples/{name}/check", status_code=202)
def check_sample(name: str, opts: SampleOptions = SampleOptions()):
    p = _sample_path(name)
    label = sample_label(p.name) or {}
    meta = store.create(p.name, p.read_bytes(),
                        _job_options(opts.use_trocr, opts.explain, opts.use_rag, label.get("reference_date"), 1),
                        sample=p.name)
    engine.submit(meta["id"])
    return meta


# ---------------------------------------------------------------------- forms

@app.post("/api/forms", status_code=202)
async def upload_form(file: UploadFile = File(...), use_trocr: bool = Form(True), explain: bool = Form(True),
                      use_rag: bool = Form(True), today: Optional[str] = Form(None), page: int = Form(1)):
    name = Path(file.filename or "form").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(415, f"'{ext or name}' files are not supported. Upload a JPG, PNG or PDF of the form.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The file is larger than {config.MAX_UPLOAD_MB:g} MB. Upload a smaller scan or photo.")
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise HTTPException(415, "This file is named .pdf but is not a PDF.")
    meta = store.create(name, data, _job_options(use_trocr, explain, use_rag, today, page))
    engine.submit(meta["id"])
    return meta


@app.get("/api/forms")
def list_forms(limit: int = Query(200, ge=1, le=1000)):
    return {"forms": store.list()[:limit]}


@app.get("/api/forms/{form_id}")
def get_form(form_id: str):
    meta = _meta_or_404(form_id)
    if meta["status"] != "done":
        return {"id": form_id, **{k: meta.get(k) for k in ("filename", "status", "stage", "progress", "error",
                                                           "created_at", "updated_at", "options", "sample")}}
    return engine.payload(form_id)


@app.delete("/api/forms/{form_id}", status_code=204)
def delete_form(form_id: str):
    meta = _meta_or_404(form_id)
    if meta["status"] in ("queued", "processing"):
        raise HTTPException(409, "This form is still being checked. Delete it when it has finished.")
    store.delete(form_id)
    return Response(status_code=204)


@app.get("/api/forms/{form_id}/image/{kind}")
def form_image(form_id: str, kind: str):
    meta = _meta_or_404(form_id)
    files = {"page": ("page.png", "image/png"), "aligned": ("aligned.png", "image/png"),
             "thumb": ("thumb.jpg", "image/jpeg")}
    if kind == "original":
        p = store.path(form_id, meta["original"])
        return FileResponse(p, filename=meta["filename"])
    if kind not in files:
        raise HTTPException(404, "Unknown image.")
    p = store.path(form_id, files[kind][0])
    if not p.exists():
        raise HTTPException(404, "This image is not ready yet.")
    return FileResponse(p, media_type=files[kind][1], headers={"Cache-Control": "no-cache"})


@app.get("/api/forms/{form_id}/fields/{name}/crop.png")
def field_crop(form_id: str, name: str):
    from formsense.summary import crop_field
    _done_or_409(form_id)
    payload = engine.payload(form_id)
    row = next((f for f in payload["fields"] if f["name"] == name), None)
    if row is None:
        raise HTTPException(404, "Unknown field.")
    img = cv2.imread(str(store.path(form_id, "aligned.png")), cv2.IMREAD_GRAYSCALE)
    return _png(crop_field(img, tuple(row["bbox"]), pad=12))


class Corrections(BaseModel):
    corrections: dict[str, dict] = {}


@app.put("/api/forms/{form_id}/corrections")
def put_corrections(form_id: str, body: Corrections):
    _done_or_409(form_id)
    try:
        return engine.run_sync(engine.apply_corrections, form_id, body.corrections)
    except ValueError as e:
        raise HTTPException(422, str(e))


class ExplainBody(BaseModel):
    use_rag: bool = True


@app.post("/api/forms/{form_id}/explain")
def explain(form_id: str, body: ExplainBody = ExplainBody()):
    _done_or_409(form_id)
    engine.run_sync(engine.explain, form_id, body.use_rag)
    return engine.payload(form_id)


@app.get("/api/forms/{form_id}/report.pdf")
def report_pdf(form_id: str):
    meta = _done_or_409(form_id)
    pdf = engine.run_sync(engine.pdf, form_id)
    stem = Path(meta["filename"]).stem
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_formsense_report.pdf"'})


@app.get("/api/forms/{form_id}/report.json")
def report_json(form_id: str):
    meta = _done_or_409(form_id)
    stem = Path(meta["filename"]).stem
    return Response(engine.report(form_id).model_dump_json(indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_formsense_report.json"'})


@app.get("/api/forms/{form_id}/checklist.csv")
def checklist_csv(form_id: str):
    meta = _done_or_409(form_id)
    stem = Path(meta["filename"]).stem
    return Response(engine.checklist_csv(form_id), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_checklist.csv"'})


# ---------------------------------------------------------------------- guidelines

@app.get("/api/guidelines")
def guidelines():
    return {"sections": engine.run_sync(engine.guideline_sections)}


@app.get("/api/guidelines/search")
def guideline_search(q: str = Query(..., min_length=2, max_length=300), k: int = Query(4, ge=1, le=10)):
    return {"query": q, "results": engine.run_sync(engine.search_guidelines, q, k)}


# ---------------------------------------------------------------------- errors + web app

@app.exception_handler(KeyError)
def _key_error(_, exc):
    return JSONResponse({"detail": "Not found."}, status_code=404)


if config.FRONTEND_DIST.joinpath("index.html").exists():
    assets = config.FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API endpoint.")
        f = (config.FRONTEND_DIST / path).resolve()
        if path and f.is_file() and config.FRONTEND_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(config.FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    @app.get("/", include_in_schema=False)
    def no_frontend():
        return JSONResponse({"message": "FormSense API is running. The web app is not built: "
                                        "see frontend/README.md. API docs: /docs"})

