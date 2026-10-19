"""Chunking strategies. Each chunk remembers its document, its section and the
document's metadata, so later steps can filter and cite."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .documents import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    section: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.doc_id} › {self.section}" if self.section else self.doc_id

    def with_context(self) -> str:
        """Text used for embedding when 'contextual chunking' is on."""
        title = self.meta.get("title", self.doc_id)
        return f"{title} › {self.section}\n{self.text}" if self.section else f"{title}\n{self.text}"


# ---------------------------------------------------------------- strategies
def fixed_size(text: str, size: int = 200, overlap: int = 0) -> list[str]:
    """Cut every `size` characters, no matter what."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive(text: str, size: int = 400, overlap: int = 0, seps=None) -> list[str]:
    """Split on the biggest natural break first (paragraph, line, sentence, word),
    only going smaller when a piece is still too big, then pack pieces up to `size`."""
    seps = SEPARATORS if seps is None else seps
    if len(text) <= size:
        return [text] if text.strip() else []
    if not seps:
        return fixed_size(text, size, overlap)
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return recursive(text, size, overlap, rest)
    pieces: list[str] = []
    for p in parts:
        pieces.extend([p] if len(p) <= size else recursive(p, size, overlap, rest))
    chunks, cur = [], ""
    for p in pieces:
        candidate = (cur + sep + p) if cur else p
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = (tail + sep + p) if tail else p
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def by_headings(text: str, max_size: int = 900) -> list[tuple[str, str]]:
    """Structure-aware: one chunk per Markdown section (split further if too long).
    Returns (section_title, text) pairs."""
    matches = list(HEADING.finditer(text))
    out: list[tuple[str, str]] = []
    if not matches:
        return [("", c) for c in recursive(text, max_size)]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        title = m.group(2).strip()
        for piece in recursive(body, max_size):
            out.append((title, piece))
    return out


def section_of(text: str, position: int) -> str:
    last = ""
    for m in HEADING.finditer(text):
        if m.start() > position:
            break
        if len(m.group(1)) > 1:
            last = m.group(2).strip()
    return last


def chunk_documents(docs: list[Document], strategy: str = "headings", size: int = 400,
                    overlap: int = 0) -> list[Chunk]:
    """strategy: 'fixed' | 'recursive' | 'headings'"""
    chunks: list[Chunk] = []
    for doc in docs:
        meta = dict(doc.meta, title=doc.title)
        if strategy == "headings":
            pairs = by_headings(doc.text, max(size, 200))
        else:
            pieces = fixed_size(doc.text, size, overlap) if strategy == "fixed" else recursive(doc.text, size, overlap)
            pairs = []
            for p in pieces:
                pos = doc.text.find(p[:40])
                pairs.append((section_of(doc.text, pos) if pos >= 0 else "", p))
        for i, (section, text) in enumerate(pairs):
            chunks.append(Chunk(f"{doc.doc_id}#{i}", doc.doc_id, text.strip(), section, meta))
    return chunks
