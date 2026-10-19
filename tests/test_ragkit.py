"""Fast unit tests for ragkit (offline, no model downloads)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("RAG_EMBEDDER", "tfidf")
os.environ.setdefault("RAG_LLM", "extractive")
os.environ.setdefault("RAG_RERANKER", "overlap")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import BM25Index, Hit, chunk_documents, load_company, load_course, rrf  # noqa: E402
from ragkit.chunking import Chunk, recursive  # noqa: E402
from ragkit.query import rule_rewrite  # noqa: E402
from ragkit.retrieval import tokenize  # noqa: E402


def test_documents_have_metadata():
    docs = load_company()
    assert len(docs) >= 9
    assert all("access" in d.meta for d in docs)
    assert {d.meta["status"] for d in docs if d.doc_id.startswith("leave")} == {"current", "superseded"}


def test_heading_chunks_keep_section():
    chunks = chunk_documents(load_course(), "headings")
    assert all(c.section for c in chunks)
    assert any("Hybrid search" in c.section for c in chunks)


def test_recursive_respects_size():
    text = "word " * 500
    assert all(len(c) <= 120 for c in recursive(text, 120))


def test_tokenize_keeps_codes():
    assert "err-5007" in tokenize("Can we retry ERR-5007?")


def test_bm25_finds_exact_code():
    chunks = chunk_documents(load_company(), "headings")
    top = BM25Index(chunks).search("Can we retry ERR-5007?", 1)[0]
    assert top.chunk.section == "ERR-5007"


def test_rrf_math():
    a, b, c = (Chunk(x, x, x) for x in "abc")
    l1 = [Hit(a, 0, 1), Hit(b, 0, 2)]
    l2 = [Hit(b, 0, 1), Hit(c, 0, 2)]
    fused = rrf(l1, l2, k=60)
    assert fused[0].chunk.id == "b"
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12


def test_rule_rewrite():
    h = ["Explain hybrid search and reranking."]
    assert rule_rewrite(h, "What are the advantages of the above?") == \
        "What are the advantages of hybrid search and reranking?"
