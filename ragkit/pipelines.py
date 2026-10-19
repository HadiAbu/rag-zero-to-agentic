"""Ready-made retrieval pipelines, one per stage of the course, so lessons 13-17
can compare them. Each returns a list of Hits (best first)."""
from __future__ import annotations

from .chunking import chunk_documents
from .documents import load_company, load_course
from .embeddings import get_embedder
from .retrieval import BM25Index, Reranker, VectorIndex, rrf


class Retriever:
    """Holds both indexes for one corpus and exposes every strategy."""

    def __init__(self, chunks, embedder=None, reranker: Reranker | None = None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        if getattr(self.embedder, "needs_fit", False):  # offline TF-IDF is fit per corpus: never share it
            self.embedder = type(self.embedder)()
        self.vec = VectorIndex(self.embedder).add(chunks)
        self.bm25 = BM25Index(chunks)
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def vector(self, q, k=3, where=None):
        return self.vec.search(q, k, where)

    def keyword(self, q, k=3, where=None):
        return self.bm25.search(q, k, where)

    def hybrid(self, q, k=3, where=None, pool=10):
        return rrf(self.vec.search(q, pool, where), self.bm25.search(q, pool, where), top=k)

    def hybrid_rerank(self, q, k=3, where=None, pool=10):
        candidates = rrf(self.vec.search(q, pool, where), self.bm25.search(q, pool, where), top=pool)
        return self.reranker.rerank(q, candidates, k)


def course_retriever(**kw) -> Retriever:
    return Retriever(chunk_documents(load_course()), **kw)


def company_retriever(**kw) -> Retriever:
    return Retriever(chunk_documents(load_company()), **kw)
