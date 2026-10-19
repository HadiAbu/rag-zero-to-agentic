"""
Lesson 14 · A real index: persistent, incremental, idempotent

PROBLEM  So far we re-embedded every document on every run. With 100,000
         documents that is slow and costs money, and running ingestion twice
         must never create duplicates.
IDEA     Store vectors on disk with a content hash per chunk. On each sync,
         embed only new or changed chunks and delete chunks that disappeared.
         Then the same idea in a real vector database (Chroma), if installed.
WATCH    "embedded" vs "reused" counts after we edit one document.

Run:  python lessons/14_vector_database.py
Optional: pip install chromadb   (pgvector and Qdrant examples: docs/VECTOR_DATABASES.md)
"""
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import chunk_documents, get_embedder, load_folder, show
from ragkit.documents import DATA
from ragkit.store import PersistentIndex

show.lesson("14", "Persistent vector store",
            "Re-embedding everything on every run is slow, costly and can duplicate data.",
            "Hash every chunk; embed only what changed; keep vectors on disk.")

embedder = get_embedder()
show.setup_info(embedder)
if getattr(embedder, "needs_fit", False):
    show.bad("The offline TF-IDF embedder can't be updated incrementally. Run with RAG_EMBEDDER=minilm.")
    show.takeaway("Incremental indexing needs a fixed embedding model (re-fitting TF-IDF changes every vector).")
    sys.exit(0)

work = pathlib.Path(tempfile.mkdtemp(prefix="rag-lesson14-"))
docs_dir, index_dir = work / "docs", work / "index"
shutil.copytree(DATA / "course", docs_dir)

show.step("1 · First sync: everything is new")
index = PersistentIndex(embedder, index_dir)
show.table([index.sync(chunk_documents(load_folder(docs_dir)))])

show.step("2 · Run it again with no changes (idempotent)")
index = PersistentIndex(embedder, index_dir)  # reload from disk, like a new process
show.table([index.sync(chunk_documents(load_folder(docs_dir)))])

show.step("3 · Edit one section of one document, then sync")
faq = docs_dir / "course-faq.md"
faq.write_text(faq.read_text().replace("room B12", "room C7"), encoding="utf-8")
show.table([index.sync(chunk_documents(load_folder(docs_dir)))])
show.hits_table(index.search("Where are office hours held?", 2), "Search after the update")

show.step("4 · The same in a real vector database (Chroma)")
try:
    import chromadb

    client = chromadb.PersistentClient(path=str(work / "chroma"))
    col = client.get_or_create_collection("course", metadata={"hnsw:space": "cosine"})
    chunks = index.chunks
    col.upsert(ids=[c.id for c in chunks], documents=[c.with_context() for c in chunks],
               embeddings=index.vectors.tolist(),
               metadatas=[{"doc_id": c.doc_id, "section": c.section} for c in chunks])
    res = col.query(query_embeddings=embedder.encode(["Where are office hours held?"]).tolist(), n_results=2,
                    where={"doc_id": "course-faq"})
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        show.info(f"   distance={dist:.3f}  {doc[:90]!r}")
    show.good("Chroma stored the same vectors, with a metadata filter (where=doc_id) and an HNSW index.")
except ImportError:
    show.info("   chromadb is not installed (pip install chromadb). See docs/VECTOR_DATABASES.md for\n"
              "   Chroma, Qdrant and pgvector versions of this lesson.")

shutil.rmtree(work, ignore_errors=True)
show.takeaway("Ingestion is a pipeline you run many times: make it idempotent and incremental.\n"
              "A vector database adds the same plus ANN indexes, filters, persistence and scale.")
