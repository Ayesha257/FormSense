"""Guideline retrieval (the "R" in RAG) with LangChain + FAISS.

    guidelines (data/guidelines/*.md)
      -> split by headings, then into ~700-character chunks
      -> embed with sentence-transformers/all-MiniLM-L6-v2
      -> FAISS index saved in data/guidelines/.index/

For each Issue the rules step finds (missing signature, bad CNIC, two programs
ticked ...), `for_issue` returns the guideline passage that explains the rule,
so the LLM explanation can quote real evidence instead of guessing.

The index rebuilds itself whenever a guidelines file changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from formsense.schema import GuidelineCitation, Issue, IssueType

BACKEND = Path(__file__).resolve().parents[2]
GUIDELINES_DIR = BACKEND / "data" / "guidelines"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 700, 80
MIN_SCORE = 0.30          # score below this = "no relevant guideline"
DENSE_WEIGHT = 0.7        # hybrid score = 0.7 * semantic + 0.3 * keyword (TF-IDF)
HEADING_BOOST = 0.2       # issues only: extra score when a section heading names the issue's field
MODES = ("dense", "tfidf", "hybrid")

# Local terms people use for the same thing (Pakistani admissions vocabulary).
# Matching words in a query get the guideline's own term appended.
SYNONYMS = {
    r"\b(id card|identity card|national id|nic|nadra card)\b": "CNIC",
    r"\bb-?form\b": "B-Form CNIC",
    r"\b(fsc|f\.sc|fa|ics|i\.com|hssc|inter|intermediate|12th)\b": "Intermediate HSSC",
    r"\b(ssc|matric|matriculation|10th)\b": "Matric SSC",
    r"\b(birthday|born|birth date)\b": "date of birth",
    r"\b(cell|cell number|phone|contact number)\b": "mobile number",
    r"\b(degree|course|major|discipline)s?\b": "program",
    r"\b(sign|signed|signing)\b": "signature",
    r"\b(age|years old)\b": "date of birth age",
}


def expand_query(query: str) -> str:
    import re
    extra = [term for rx, term in SYNONYMS.items() if re.search(rx, query, re.IGNORECASE)]
    return f"{query} {' '.join(extra)}" if extra else query


@dataclass
class Passage:
    text: str
    source: str           # file name
    section: str          # e.g. "Section D: Declaration > 6.3 Applicant Signature"
    score: float          # cosine similarity, 0..1 (higher = more relevant)

    @property
    def ref(self) -> str:
        return f"{self.source} § {self.section}" if self.section else self.source


# ---------------- loading and chunking ----------------

def guideline_files(folder: Path = GUIDELINES_DIR) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".md", ".txt"))


def load_chunks(folder: Path = GUIDELINES_DIR):
    """Guideline files -> LangChain Documents, one per section chunk."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for path in guideline_files(folder):
        text = path.read_text(encoding="utf-8")
        sections = by_heading.split_text(text) if path.suffix.lower() == ".md" else None
        if not sections:  # plain text file: size-based chunks only
            from langchain_core.documents import Document
            sections = [Document(page_content=text, metadata={})]
        for d in by_size.split_documents(sections):
            body = "\n".join(l for l in d.page_content.splitlines()
                             if not l.lstrip().startswith("#")).strip()
            if not body or body.startswith(">"):        # skip headings-only and notes like the SAMPLE banner
                continue
            section = " > ".join(d.metadata[k] for k in ("h2", "h3") if d.metadata.get(k))
            d.metadata = {"source": path.name, "section": section or d.metadata.get("h1", "")}
            docs.append(d)
    return docs


def fingerprint(folder: Path = GUIDELINES_DIR) -> str:
    h = hashlib.sha256(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for p in guideline_files(folder):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------- retriever ----------------

class GuidelineRetriever:
    def __init__(self, folder: Path | str = GUIDELINES_DIR, index_dir: Path | str | None = None,
                 embeddings=None, mode: str = "hybrid", expand: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.folder = Path(folder)
        self.index_dir = Path(index_dir) if index_dir else self.folder / ".index"
        self.mode = mode
        self.expand = expand
        self._embeddings = embeddings
        self._store = None
        self._docs = None        # chunks in index order
        self._tfidf = None       # (vectorizer, matrix)

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
        return self._embeddings

    # --- index ---
    def build(self) -> int:
        from langchain_community.vectorstores import FAISS
        docs = load_chunks(self.folder)
        if not docs:
            raise FileNotFoundError(f"No guideline text found in {self.folder} (.md or .txt)")
        self._store = FAISS.from_documents(docs, self.embeddings)
        self._docs, self._tfidf = None, None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_dir))
        (self.index_dir / "manifest.json").write_text(json.dumps(
            {"fingerprint": fingerprint(self.folder), "chunks": len(docs),
             "model": EMBED_MODEL, "files": [p.name for p in guideline_files(self.folder)]}, indent=2))
        return len(docs)

    def is_fresh(self) -> bool:
        m = self.index_dir / "manifest.json"
        return m.exists() and json.loads(m.read_text()).get("fingerprint") == fingerprint(self.folder)

    def ensure(self):
        """Load the saved index, rebuilding it first if the guidelines changed."""
        if self._store is not None and self.is_fresh():
            return self
        if not self.is_fresh():
            self.build()
            return self
        from langchain_community.vectorstores import FAISS
        # the index is a file we built ourselves, so loading its pickle is safe
        self._store = FAISS.load_local(str(self.index_dir), self.embeddings,
                                       allow_dangerous_deserialization=True)
        self._docs, self._tfidf = None, None
        return self

    def _keyword_index(self):
        if self._tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            ids = self._store.index_to_docstore_id
            self._docs = [self._store.docstore.search(ids[i]) for i in range(len(ids))]
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self._tfidf = (vec, vec.fit_transform(d.page_content for d in self._docs))
        return self._tfidf

    # --- search ---
    def search(self, query: str, k: int = 3, mode: str | None = None,
               expand: bool | None = None) -> list[Passage]:
        """Top-k guideline chunks. mode: dense (embeddings), tfidf (keywords) or hybrid (both)."""
        import numpy as np
        self.ensure()
        mode = mode or self.mode
        if self.expand if expand is None else expand:
            query = expand_query(query)
        n = self._store.index.ntotal
        vec, mat = self._keyword_index()
        dense = np.zeros(n)
        if mode != "tfidf":
            q = np.asarray(self.embeddings.embed_query(query), dtype="float32")[None, :]
            dist, idx = self._store.index.search(q, n)
            # squared L2 between unit vectors -> cosine similarity: cos = 1 - d / 2
            dense[idx[0]] = np.clip(1.0 - dist[0] / 2.0, 0.0, 1.0)
        keyword = (mat @ vec.transform([query]).T).toarray().ravel() if mode != "dense" else np.zeros(n)
        score = {"dense": dense, "tfidf": keyword,
                 "hybrid": DENSE_WEIGHT * dense + (1 - DENSE_WEIGHT) * keyword}[mode]
        out = []
        for i in np.argsort(-score)[:k]:
            d = self._docs[i]
            out.append(Passage(d.page_content, d.metadata.get("source", ""),
                               d.metadata.get("section", ""), round(float(score[i]), 3)))
        return out

    def search_for_issue(self, issue: Issue, labels: dict[str, str] | None = None, k: int = 1,
                         mode: str | None = None, expand: bool | None = None,
                         heading_boost: float = HEADING_BOOST) -> list[Passage]:
        """Like search(), but an issue names its fields, so a section whose heading
        matches a field's label (e.g. "2.4 Date of Birth") gets a small boost."""
        pool = self.search(query_for_issue(issue, labels), k=max(k, 8), mode=mode, expand=expand)
        names = [(labels or {}).get(f, f.replace("_", " ")) for f in issue.fields]
        targets = [t for t in (_tokens(n) for n in names) if t]
        if heading_boost and targets:
            for p in pool:
                heading = _tokens(p.section.split(">")[-1])
                best = max(len(t & heading) / len(t) for t in targets)
                p.score = round(p.score + heading_boost * best, 3)
            pool.sort(key=lambda p: -p.score)
        return pool[:k]

    def for_issue(self, issue: Issue, labels: dict[str, str] | None = None, k: int = 1,
                  min_score: float = MIN_SCORE) -> list[GuidelineCitation]:
        """Best guideline passage(s) for one Issue, as citations for the report."""
        return [GuidelineCitation(issue_id=issue.id, text=p.text, source=p.ref)
                for p in self.search_for_issue(issue, labels, k=k) if p.score >= min_score]


_STOP = {"no", "of", "the", "and", "for", "a", "an", "in", "to", "s", "number"}


def _tokens(text: str) -> set[str]:
    """Lower-case content words with a crude plural strip ('marks' -> 'mark')."""
    import re
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and not w.isdigit()}


def field_labels(template_path: Path | str | None = None) -> dict[str, str]:
    """{field name: how the guidelines talk about it}. Checkboxes use their group
    ('Gender', 'Program') because rules about them are written per group."""
    path = Path(template_path) if template_path else BACKEND / "data" / "templates" / "admission_form.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups") or {}
    out = {}
    for f in data["fields"]:
        g = f.get("group")
        if f.get("type") == "checkbox" and g:
            out[f["name"]] = (groups.get(g) or {}).get("label") or g.replace("_", " ").title()
        else:
            out[f["name"]] = f["label"]
    return out


def query_for_issue(issue: Issue, labels: dict[str, str] | None = None) -> str:
    """Turn an Issue into a natural-language search query.

    Field names in the message ('matric_obtained') are swapped for the printed
    labels ('Matric Obtained Marks'), because the guidelines use the labels.
    """
    import re
    labels = labels or {}
    names = [labels.get(f, f.replace("_", " ")) for f in issue.fields]
    what = " and ".join(dict.fromkeys(names)) if names else "form"
    lead = {
        IssueType.MISSING: f"{what} is required and was left blank",
        IssueType.UNREADABLE: f"{what} is unclear, write clearly in block letters inside the box",
        IssueType.FORMAT: f"correct format for {what}",
        IssueType.INCONSISTENCY: f"{what} must match and be consistent",
        IssueType.SELECTION: f"how many boxes to tick for {what}",
    }[issue.type]
    message = issue.message
    for name in sorted(set(issue.fields) | set(labels), key=len, reverse=True):
        if name in labels:
            message = re.sub(rf"\b{re.escape(name)}\b", labels[name], message)
    return f"{lead}. {message}"
