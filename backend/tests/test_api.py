"""Web API smoke tests: the routes the frontend relies on, without loading the OCR models."""
import os
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")   # FastAPI's TestClient needs it

os.environ.setdefault("FORMSENSE_WARM_UP", "0")
os.environ.setdefault("FORMSENSE_HISTORY_DIR", tempfile.mkdtemp(prefix="fs_history_"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_limits(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["max_upload_mb"] > 0
    assert ".pdf" in body["allowed"]


def test_template_lists_fields(client):
    r = client.get("/api/template")
    assert r.status_code == 200
    t = r.json()
    assert t["width"] > 0 and t["height"] > 0
    assert len(t["fields"]) > 10


def test_upload_rejects_wrong_type_and_empty_files(client):
    r = client.post("/api/forms", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/forms", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/forms", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 415


def test_unknown_form_is_404(client):
    assert client.get("/api/forms/abcdef123456").status_code == 404
    assert client.get("/api/forms/..%2f..%2fetc").status_code == 404
    assert client.get("/api/nope").status_code == 404


def test_history_and_stats_start_empty(client):
    assert client.get("/api/forms").json()["forms"] == []
    s = client.get("/api/stats").json()
    assert s["total"] == 0 and len(s["days"]) == 7
