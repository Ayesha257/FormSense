"""Map a printed label ("D.O.B", "Father's Name", "Mobile No.") to a canonical field name.

Why this exists
---------------
After template alignment the field boxes are known by position, so most fields need no
mapping. The mapper is the safety net for the cases where position is not enough: the
label text was read by OCR, a different template version prints different wording, or
alignment was poor and the page has to be understood from its text.

Stages (cheapest first, each one only runs if the previous found nothing)
-----------------------------------------------------------------------
1. exact  : normalised label equals the field's template label, its name as words, or a
             listed alias. Instant and unambiguous.
2. fuzzy  : rapidfuzz token sort ratio against all of those texts. Survives OCR typos.
3. embed  : sentence-transformers MiniLM cosine similarity. Survives different wording
             ("Date you were born"). Loaded lazily; if it is not installed or cannot be
             downloaded the mapper keeps working with stages 1 and 2 and says so once.

The mapper never guesses: if the best match is weak or too close to the runner up it
returns ``field=None`` so the caller can route the label to review.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml
from rapidfuzz import fuzz

from formsense.nlp.template_info import TemplateInfo

log = logging.getLogger("formsense.nlp")

DEFAULT_ALIASES_PATH = Path(__file__).with_name("label_aliases.yaml")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EmbedFn = Callable[[Sequence[str]], "object"]      # texts -> array of shape (n, d)


@dataclass(frozen=True)
class MapResult:
    field: Optional[str]
    score: float                    # 0..1
    method: str                     # exact | alias | fuzzy | embedding | ambiguous | none
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0


def norm_label(text: str) -> str:
    """Lowercase, drop dots and apostrophes (D.O.B -> dob), punctuation to spaces."""
    s = (text or "").lower().replace("%", " percent ")
    s = re.sub(r"[.'`\u2019]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_FOLD = str.maketrans("l10", "iio")


def _fold(s: str) -> str:
    """Collapse look-alike characters (l/1/i, 0/o) so OCR slips such as 'CNlC' still match."""
    return s.translate(_FOLD)


def load_aliases(path: Union[str, Path, None] = None) -> Dict[str, List[str]]:
    p = Path(path) if path else DEFAULT_ALIASES_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {k: [str(v) for v in (vals or [])] for k, vals in data.items()}


def default_embed_fn(model_name: str = DEFAULT_MODEL) -> EmbedFn:
    from sentence_transformers import SentenceTransformer  # heavy; imported on demand
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


class FieldMapper:
    def __init__(
        self,
        template: Union[TemplateInfo, Mapping, str, Path, None] = None,
        aliases: Optional[Mapping[str, Sequence[str]]] = None,
        embed_fn: Optional[EmbedFn] = None,
        use_embeddings: bool = True,
        fuzzy_threshold: float = 0.86,
        embed_threshold: float = 0.60,
        margin: float = 0.04,
    ):
        self.tpl = TemplateInfo.coerce(template)
        alias_map = load_aliases() if aliases is None else {k: list(v) for k, v in aliases.items()}
        self.fuzzy_threshold = fuzzy_threshold
        self.embed_threshold = embed_threshold
        self.margin = margin
        self._embed_fn = embed_fn
        self._use_embeddings = use_embeddings
        self._warned = False

        # candidates: list of (field, normalised text, kind) where kind is "exact" or "alias"
        self._cands: List[Tuple[str, str, str]] = []
        for name, info in self.tpl.fields.items():
            for text, kind in ((info.label, "exact"), (name.replace("_", " "), "exact")):
                self._add(name, text, kind)
            for a in alias_map.get(name, []):
                self._add(name, a, "alias")
        unknown = [k for k in alias_map if k not in self.tpl.fields]
        if unknown:
            log.debug("aliases for fields not in this template are ignored: %s", unknown)
        self._exact_index: Dict[str, List[Tuple[str, str]]] = {}
        for f, t, kind in self._cands:
            self._exact_index.setdefault(t, []).append((f, kind))
        self._cand_vecs = None          # built lazily when embeddings are first needed

    # ---------- public API ----------
    def map_label(self, text: str) -> MapResult:
        cands = self._candidates(text)
        if not cands:
            return MapResult(None, 0.0, "none")
        best = cands[0]
        runner = next((c for c in cands[1:] if c[0] != best[0]), None)
        r_field, r_score = (runner[0], runner[1]) if runner else (None, 0.0)
        if best[2] in ("fuzzy", "embedding") and runner and best[1] - r_score < self.margin:
            return MapResult(None, best[1], "ambiguous", r_field, r_score)
        if best[2] in ("exact", "alias") and runner and runner[2] in ("exact", "alias") and runner[1] >= best[1]:
            return MapResult(None, best[1], "ambiguous", r_field, r_score)
        return MapResult(best[0], best[1], best[2], r_field, r_score)

    def map_labels(self, texts: Sequence[str]) -> List[MapResult]:
        return [self.map_label(t) for t in texts]

    def map_labels_one_to_one(self, texts: Sequence[str]) -> List[MapResult]:
        """Assign each printed label to a different field (greedy by score).

        Use this when you have all the labels of one page: a field can only be claimed
        once, which removes most remaining ambiguity.
        """
        options: List[Tuple[float, int, str, str]] = []
        for i, t in enumerate(texts):
            for f, s, m in self._candidates(t):
                options.append((s, i, f, m))
        # best score first; on ties the longer (more specific) label claims the field first
        options.sort(key=lambda o: (-o[0], -len(norm_label(texts[o[1]])), o[1]))
        out: List[MapResult] = [MapResult(None, 0.0, "none") for _ in texts]
        used_label, used_field = set(), set()
        for s, i, f, m in options:
            if i in used_label or f in used_field:
                continue
            out[i] = MapResult(f, s, m)
            used_label.add(i)
            used_field.add(f)
        return out

    # ---------- internals ----------
    def _add(self, field: str, text: str, kind: str) -> None:
        n = norm_label(text)
        if n and not any(c[0] == field and c[1] == n for c in self._cands):
            self._cands.append((field, n, kind))

    def _candidates(self, text: str) -> List[Tuple[str, float, str]]:
        """Eligible (field, score, method) triples, best first."""
        q = norm_label(text)
        if not q:
            return []
        hits = self._exact_index.get(q)
        if hits:
            out = {}
            for f, kind in hits:
                prev = out.get(f)
                out[f] = "exact" if (prev == "exact" or kind == "exact") else "alias"
            return sorted(((f, 1.0, m) for f, m in out.items()), key=lambda x: (x[2] != "exact", x[0]))

        by_field: Dict[str, float] = {}
        for f, t, _ in self._cands:
            by_field[f] = max(by_field.get(f, 0.0), fuzz.token_sort_ratio(_fold(q), _fold(t)) / 100.0)
        fuzzy = sorted(((f, s, "fuzzy") for f, s in by_field.items() if s >= self.fuzzy_threshold),
                       key=lambda x: -x[1])
        if fuzzy:
            return fuzzy
        return self._embedding_candidates(q)

    def _embedding_candidates(self, q: str) -> List[Tuple[str, float, str]]:
        if not self._use_embeddings:
            return []
        try:
            import numpy as np
            if self._embed_fn is None:
                self._embed_fn = default_embed_fn()
            if self._cand_vecs is None:
                self._cand_vecs = _unit(np.asarray(self._embed_fn([t for _, t, _ in self._cands])))
            v = _unit(np.asarray(self._embed_fn([q])))[0]
        except Exception as e:     # missing package, no network for the model download, ...
            if not self._warned:
                log.warning("embedding stage disabled (%s); using exact and fuzzy matching only", type(e).__name__)
                self._warned = True
            self._use_embeddings = False
            return []
        sims = self._cand_vecs @ v
        best: Dict[str, float] = {}
        for (f, _, _), s in zip(self._cands, sims):
            best[f] = max(best.get(f, -1.0), float(s))
        return sorted(((f, s, "embedding") for f, s in best.items() if s >= self.embed_threshold),
                      key=lambda x: -x[1])


def _unit(a):
    import numpy as np
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n == 0, 1, n)
