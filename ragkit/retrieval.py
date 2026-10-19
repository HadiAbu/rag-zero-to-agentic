"""Retrieval building blocks: a vector index, a BM25 keyword index, reciprocal
rank fusion and a reranker. Everything returns a list of `Hit`s, best first."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import Chunk

Filter = Callable[[Chunk], bool]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0
    details: dict = field(default_factory=dict)


def _ranked(chunks, scores, k, where: Filter | None) -> list[Hit]:
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = []
    for i in order:
        c = chunks[i]
        if where is not None and not where(c):
            continue
        hits.append(Hit(c, float(scores[i])))
        if len(hits) >= k:
            break
    for r, h in enumerate(hits, 1):
        h.rank = r
    return hits


# ---------------------------------------------------------------- vectors
class VectorIndex:
    """A tiny in-memory vector store: exact cosine search over normalized vectors."""

    def __init__(self, embedder, contextual: bool = True):
        self.embedder = embedder
        self.contextual = contextual
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def _texts(self, chunks):
        return [c.with_context() if self.contextual else c.text for c in chunks]

    def add(self, chunks: list[Chunk]) -> "VectorIndex":
        self.chunks = list(self.chunks) + list(chunks)
        texts = self._texts(self.chunks)
        if getattr(self.embedder, "needs_fit", False):
            self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)
        return self

    def search(self, query: str, k: int = 3, where: Filter | None = None) -> list[Hit]:
        q = self.embedder.encode([query])[0]
        return _ranked(self.chunks, self.vectors @ q, k, where)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embedder.encode([a, b])
        return float(va @ vb)


# ---------------------------------------------------------------- BM25
STOP = set("a about an and are as at be but by can do does for from has have how i if in is it its "
           "me my of on or our should so that the their them then there these they this to us "
           "was we what when where which who why will with would you your".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words; keeps codes like 'err-5003' and 'pol-204' as single tokens."""
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOP]


class BM25Index:
    """Classic BM25 keyword ranking, written out so you can read every term."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, contextual: bool = True):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c.with_context() if contextual else c.text) for c in chunks]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = float(self.lengths.mean()) if len(self.docs) else 1.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            for t in q:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    s[i] += self.idf[t] * f * (self.k1 + 1) / norm
        return s

    def search(self, query: str, k: int = 3, where: Filter | None = None) -> list[Hit]:
        s = self.scores(query)
        hits = _ranked(self.chunks, s, k, where)
        return [h for h in hits if h.score > 0]


# ---------------------------------------------------------------- fusion
def rrf(*ranked_lists: list[Hit], k: int = 60, top: int = 5) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    parts: dict[str, dict] = {}
    by_id: dict[str, Chunk] = {}
    for li, hits in enumerate(ranked_lists):
        for h in hits:
            cid = h.chunk.id
            by_id[cid] = h.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + h.rank)
            parts.setdefault(cid, {})[f"list{li + 1}_rank"] = h.rank
    order = sorted(scores, key=lambda c: -scores[c])[:top]
    return [Hit(by_id[c], scores[c], r, parts[c]) for r, c in enumerate(order, 1)]


# ---------------------------------------------------------------- reranking
_CE: dict = {}


class Reranker:
    """Cross-encoder reranker (free, ~90 MB). Falls back to word overlap offline.

    RAG_RERANKER=ms-marco (default) | bge | overlap
    """

    MODELS = {"ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bge": "BAAI/bge-reranker-base"}

    def __init__(self, name: str | None = None):
        import os

        name = (name or os.getenv("RAG_RERANKER", "ms-marco")).lower()
        self.model = None
        self.name = "word-overlap (offline)"
        if name != "overlap" and os.getenv("RAG_EMBEDDER", "") != "tfidf":
            try:
                from sentence_transformers import CrossEncoder

                model_name = self.MODELS.get(name, name)
                if model_name not in _CE:
                    _CE[model_name] = CrossEncoder(model_name)
                self.model = _CE[model_name]
                self.name = model_name
            except Exception:  # noqa: BLE001
                self.model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self.model is not None:
            return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]

    def rerank(self, query: str, hits: list[Hit], top: int = 3) -> list[Hit]:
        scores = self.score(query, [h.chunk.with_context() for h in hits])
        out = []
        for h, s in zip(hits, scores):
            out.append(Hit(h.chunk, s, 0, {"before": h.rank}))
        out.sort(key=lambda h: -h.score)
        for r, h in enumerate(out, 1):
            h.rank = r
        return out[:top]


# ---------------------------------------------------------------- persistence
def save_index(index: VectorIndex, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", index.vectors)
    rows = [{"id": c.id, "doc_id": c.doc_id, "text": c.text, "section": c.section, "meta": c.meta} for c in index.chunks]
    (folder / "chunks.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def load_index(embedder, folder: str | Path) -> VectorIndex:
    folder = Path(folder)
    idx = VectorIndex(embedder)
    rows = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    idx.chunks = [Chunk(r["id"], r["doc_id"], r["text"], r["section"], r["meta"]) for r in rows]
    idx.vectors = np.load(folder / "vectors.npy")
    return idx
