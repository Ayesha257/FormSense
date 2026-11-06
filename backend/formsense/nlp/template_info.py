"""A small, forgiving view of the form template.

Both the extractor (nlp/) and the rule engine (rules/) need to know, for each field,
its type, whether it is required, and which checkbox group it belongs to. This module
reads that from the template JSON once, so no other module hardcodes field lists.

The JSON may be the plain ``FormTemplate`` shape from schema.py or the richer shape
written by the generator (extra keys such as ``kind`` and ``groups`` are used when
present and inferred when absent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

# backend/formsense/nlp/template_info.py -> parents[2] is the backend folder
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = BACKEND_ROOT / "data" / "templates" / "admission_form.json"

_NUMBER_SUFFIXES = ("_total", "_obtained", "_percentage")


@dataclass(frozen=True)
class FieldInfo:
    name: str
    label: str
    type: str                      # "text" | "date" | "checkbox" | "signature"
    kind: str                      # "text" | "date" | "number" | "checkbox" | "signature" (or any custom value)
    required: bool
    group: Optional[str]
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class TemplateInfo:
    name: str
    fields: Dict[str, FieldInfo]               # insertion order = template order
    groups: Dict[str, List[str]]               # group name -> member field names (template order)

    # ----- constructors -----
    @classmethod
    def from_dict(cls, tpl: Mapping[str, Any]) -> "TemplateInfo":
        fields: Dict[str, FieldInfo] = {}
        for raw in tpl["fields"]:
            name = raw["name"]
            ftype = _enum_value(raw.get("type", "text"))
            fields[name] = FieldInfo(
                name=name,
                label=raw.get("label") or name.replace("_", " ").title(),
                type=ftype,
                kind=_infer_kind(name, ftype, raw.get("kind")),
                required=bool(raw.get("required", True)),
                group=raw.get("group"),
                bbox=tuple(int(v) for v in raw.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
            )
        groups: Dict[str, List[str]] = {g: [] for g in (tpl.get("groups") or {})}
        for f in fields.values():
            if f.group:
                groups.setdefault(f.group, []).append(f.name)
        return cls(name=str(tpl.get("name", "form")), fields=fields, groups=groups)

    @classmethod
    def from_model(cls, model: Any) -> "TemplateInfo":
        """Accept a schema.FormTemplate instance."""
        return cls.from_dict(model.model_dump() if hasattr(model, "model_dump") else model.dict())

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "TemplateInfo":
        p = Path(path) if path else DEFAULT_TEMPLATE_PATH
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def coerce(cls, tpl: Union["TemplateInfo", Mapping[str, Any], str, Path, None]) -> "TemplateInfo":
        if isinstance(tpl, TemplateInfo):
            return tpl
        if tpl is None or isinstance(tpl, (str, Path)):
            return cls.load(tpl)
        if isinstance(tpl, Mapping):
            return cls.from_dict(tpl)
        return cls.from_model(tpl)

    # ----- helpers -----
    def label(self, name: str) -> str:
        info = self.fields.get(name)
        return info.label if info else name

    def bbox(self, name: str) -> Optional[Tuple[int, int, int, int]]:
        info = self.fields.get(name)
        return info.bbox if info else None


def _enum_value(v: Any) -> str:
    return str(getattr(v, "value", v))


def _infer_kind(name: str, ftype: str, kind: Optional[str]) -> str:
    if kind:
        return str(kind)
    if ftype in ("date", "checkbox", "signature"):
        return ftype
    if name.endswith(_NUMBER_SUFFIXES):
        return "number"
    return "text"
