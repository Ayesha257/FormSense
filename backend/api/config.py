"""Paths and limits for the web API. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent

TEMPLATE = BACKEND / "data" / "templates" / "admission_form.json"
HISTORY_DIR = Path(os.getenv("FORMSENSE_HISTORY_DIR", BACKEND / "data" / "history"))
SAMPLE_DIR = Path(os.getenv("FORMSENSE_SAMPLE_DIR", BACKEND / "data" / "synthetic" / "dev"))
FRONTEND_DIST = Path(os.getenv("FORMSENSE_FRONTEND_DIST", REPO / "frontend" / "dist"))

MAX_UPLOAD_MB = float(os.getenv("FORMSENSE_MAX_UPLOAD_MB", "20"))
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SAMPLES = int(os.getenv("FORMSENSE_MAX_SAMPLES", "24"))
WARM_UP = os.getenv("FORMSENSE_WARM_UP", "1") != "0"   # load the models when the server starts
