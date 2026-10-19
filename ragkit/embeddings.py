"""Embedding models.

Default: a free local Sentence-Transformers model (all-MiniLM-L6-v2, ~90 MB, downloaded once).
Offline fallback: TF-IDF + LSA, which needs no download but only knows the words in your
documents, so it cannot match synonyms. Choose with the RAG_EMBEDDER environment variable:

    RAG_EMBEDDER=minilm   (default)  sentence-transformers/all-MiniLM-L6-v2
    RAG_EMBEDDER=bge      BAAI/bge-small-en-v1.5
    RAG_EMBEDDER=multi    paraphrase-multilingual-MiniLM-L12-v2 (Arabic, Hebrew, ...)
    RAG_EMBEDDER=tfidf    offline, no download
"""
from __future__ import annotations

import os
import warnings

import numpy as np

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "multi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
_CACHE: dict = {}


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class SentenceTransformerEmbedder:
    needs_fit = False

    def __init__(self, model_name: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        if model_name not in _CACHE:
            _CACHE[model_name] = SentenceTransformer(model_name)
        self.model = _CACHE[model_name]
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class TfidfEmbedder:
    """Offline stand-in. Learns word statistics from the corpus you fit it on."""

    needs_fit = True
    name = "tfidf-lsa (offline)"

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.vec = None
        self.svd = None

    def fit(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w[\w-]*\b")
        x = self.vec.fit_transform(texts)
        n = min(self.dims, x.shape[0] - 1, x.shape[1] - 1)
        self.svd = TruncatedSVD(n, random_state=0).fit(x) if n >= 2 else None
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self.vec.transform(list(texts))
        z = self.svd.transform(x) if self.svd is not None else x.toarray()
        return _normalize(np.asarray(z, dtype=np.float32))


def get_embedder(name: str | None = None):
    name = (name or os.getenv("RAG_EMBEDDER", "minilm")).lower()
    if name == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(MODELS.get(name, name))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not load embedding model ({exc}). Falling back to offline TF-IDF.")
        return TfidfEmbedder()
