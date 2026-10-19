"""Load Markdown documents with a small YAML-like front-matter header.

Every file in data/ starts with:

    ---
    title: ...
    doc_id: ...
    access: internal
    ---

The header becomes `Document.meta`, which later lessons use for filtering,
versioning and access control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    path: str
    meta: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta: dict = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if "," in value and key.strip() == "allowed_roles":
                meta[key.strip()] = [v.strip() for v in value.split(",")]
            else:
                meta[key.strip()] = value
    body = raw[end + 4:].lstrip("\n")
    return meta, body


def load_folder(folder: str | Path) -> list[Document]:
    """Load every .md file under `folder` (recursively), sorted by path."""
    folder = Path(folder)
    if not folder.is_absolute():
        folder = DATA / folder
    docs = []
    for path in sorted(folder.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("doc_id", path.stem)
        try:
            meta.setdefault("source", str(path.relative_to(DATA)))
        except ValueError:
            meta.setdefault("source", path.name)
        docs.append(Document(doc_id, meta.get("title", path.stem), body, str(path), meta))
    return docs


def load_course() -> list[Document]:
    return load_folder("course")


def load_company() -> list[Document]:
    return load_folder("company")
