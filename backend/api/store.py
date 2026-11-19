"""Processing history on disk: one folder per checked form.

    data/history/<id>/
        meta.json          status, stage, file name, options, timings (small, listed often)
        result.json        everything the result page shows
        vision.json        raw vision readings, kept so corrections can be re-checked
        original.<ext>     the uploaded file
        page.png           the page that was checked (first page of a PDF by default)
        aligned.png        that page aligned to the blank template
        thumb.jpg          small preview for lists

The folder is git-ignored: uploads can contain CNIC and phone numbers.
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api import config

_ID_CHARS = set("0123456789abcdef")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_id(form_id: str) -> bool:
    return 8 <= len(form_id) <= 32 and set(form_id) <= _ID_CHARS


class HistoryStore:
    def __init__(self, root: Path = config.HISTORY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- paths ----
    def folder(self, form_id: str) -> Path:
        if not valid_id(form_id):
            raise KeyError(form_id)
        return self.root / form_id

    def path(self, form_id: str, name: str) -> Path:
        return self.folder(form_id) / name

    # ---- create / read / write ----
    def create(self, filename: str, data: bytes, options: dict, sample: Optional[str] = None) -> dict:
        form_id = uuid.uuid4().hex[:12]
        folder = self.folder(form_id)
        folder.mkdir(parents=True)
        ext = Path(filename).suffix.lower() or ".bin"
        (folder / f"original{ext}").write_bytes(data)
        meta = {"id": form_id, "filename": filename, "original": f"original{ext}", "size": len(data),
                "created_at": now_iso(), "updated_at": now_iso(), "status": "queued", "stage": "queued",
                "progress": 0.02, "error": None, "options": options, "sample": sample, "summary": None}
        self.write_meta(form_id, meta)
        return meta

    def exists(self, form_id: str) -> bool:
        try:
            return (self.folder(form_id) / "meta.json").exists()
        except KeyError:
            return False

    def read_json(self, form_id: str, name: str) -> Optional[Any]:
        p = self.path(form_id, name)
        if not p.exists():
            return None
        with self._lock:
            return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, form_id: str, name: str, data: Any) -> None:
        p = self.path(form_id, name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)

    def meta(self, form_id: str) -> Optional[dict]:
        return self.read_json(form_id, "meta.json")

    def write_meta(self, form_id: str, meta: dict) -> None:
        self.write_json(form_id, "meta.json", meta)

    def update_meta(self, form_id: str, **changes) -> dict:
        with self._lock:
            meta = self.meta(form_id) or {}
            meta.update(changes)
            meta["updated_at"] = now_iso()
            self.write_meta(form_id, meta)
            return meta

    def original_bytes(self, form_id: str) -> tuple[bytes, str]:
        meta = self.meta(form_id)
        return self.path(form_id, meta["original"]).read_bytes(), meta["filename"]

    # ---- listing ----
    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir() if self.root.exists() else []:
            if d.is_dir() and valid_id(d.name) and (d / "meta.json").exists():
                try:
                    out.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)

    def delete(self, form_id: str) -> bool:
        folder = self.folder(form_id)
        if not folder.exists():
            return False
        with self._lock:
            shutil.rmtree(folder)
        return True

    def mark_interrupted(self) -> None:
        """Jobs that were running when the server stopped will never finish: say so."""
        for meta in self.list():
            if meta.get("status") in ("queued", "processing"):
                self.update_meta(meta["id"], status="failed", stage="failed",
                                 error="The server stopped before this form was checked. Check it again.")
