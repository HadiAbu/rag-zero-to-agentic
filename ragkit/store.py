"""A persistent vector store with incremental updates.

Production ingestion must be:
* idempotent  - running it twice changes nothing
* incremental - only new or changed chunks are re-embedded (embedding costs time/money)

We give each chunk a content hash. On every sync we embed only chunks whose hash is new,
and delete chunks that disappeared. Vectors and chunks are saved to disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .retrieval import VectorIndex, _ranked


def chunk_hash(chunk: Chunk, model_name: str) -> str:
    return hashlib.sha1((model_name + "|" + chunk.with_context()).encode("utf-8")).hexdigest()


class PersistentIndex(VectorIndex):
    def __init__(self, embedder, folder: str | Path):
        super().__init__(embedder, contextual=True)
        self.folder = Path(folder)
        self.hashes: list[str] = []
        if (self.folder / "manifest.json").exists():
            self._load()

    def _load(self):
        rows = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(r["id"], r["doc_id"], r["text"], r["section"], r["meta"]) for r in rows]
        self.hashes = [r["hash"] for r in rows]
        self.vectors = np.load(self.folder / "vectors.npy")

    def _save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        rows = [{"id": c.id, "doc_id": c.doc_id, "text": c.text, "section": c.section, "meta": c.meta, "hash": h}
                for c, h in zip(self.chunks, self.hashes)]
        (self.folder / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(self.folder / "vectors.npy", self.vectors)

    def sync(self, chunks: list[Chunk]) -> dict:
        if getattr(self.embedder, "needs_fit", False):
            raise RuntimeError("The offline TF-IDF embedder must be refit on the whole corpus; "
                               "use a sentence-transformers model for incremental indexing.")
        name = getattr(self.embedder, "name", "model")
        old = {h: i for i, h in enumerate(self.hashes)}
        new_hashes = [chunk_hash(c, name) for c in chunks]
        to_embed = [i for i, h in enumerate(new_hashes) if h not in old]
        vectors = np.zeros((len(chunks), self.vectors.shape[1] if self.vectors.size else 0), dtype=np.float32)
        if to_embed:
            fresh = self.embedder.encode([chunks[i].with_context() for i in to_embed])
            if vectors.shape[1] == 0:
                vectors = np.zeros((len(chunks), fresh.shape[1]), dtype=np.float32)
            for j, i in enumerate(to_embed):
                vectors[i] = fresh[j]
        for i, h in enumerate(new_hashes):
            if h in old:
                vectors[i] = self.vectors[old[h]]
        stats = {"embedded": len(to_embed), "reused": len(chunks) - len(to_embed),
                 "removed": len(set(self.hashes) - set(new_hashes))}
        self.chunks, self.hashes, self.vectors = list(chunks), new_hashes, vectors
        self._save()
        return stats

    def search(self, query: str, k: int = 3, where=None):
        q = self.embedder.encode([query])[0]
        return _ranked(self.chunks, self.vectors @ q, k, where)
