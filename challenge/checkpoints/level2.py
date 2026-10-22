"""
CHECKPOINT · levels 1–2 done: ingestion and the Chroma index.
Behind? Copy it over your file:   cp challenge/checkpoints/level2.py challenge/pipeline.py
then press "Reload pipeline" in the app.
"""
import pathlib
import re

import networkx as nx

from helpers import embed

HERE = pathlib.Path(__file__).resolve().parent if pathlib.Path(__file__).parent.name == "challenge" else pathlib.Path(__file__).resolve().parents[1]
K = 3
CHUNK_SIZE, OVERLAP = 800, 100
REFUSAL = "I don't know based on the CVs."
SYSTEM_PROMPT = ("You are a recruiting assistant. Answer ONLY from the numbered CV passages. "
                 "Cite every fact like [1]. If the passages do not contain the answer, reply exactly: "
                 f"\"{REFUSAL}\" Be brief.")


# ================================================================= LEVEL 1 · INGEST
def read_file(path):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if path.suffix.lower() == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_cvs(folder):
    files = sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in (".pdf", ".docx", ".txt", ".md"))
    return [{"source": p.name, "text": read_file(p)} for p in files]


def chunk(doc):
    text, out, start = re.sub(r"\s+", " ", doc["text"]).strip(), [], 0
    while start < len(text):
        out.append({"id": f"{doc['source']}#{len(out)}", "source": doc["source"], "text": text[start:start + CHUNK_SIZE]})
        start += CHUNK_SIZE - OVERLAP
    return out


# ================================================================= LEVEL 2 · INDEX (Chroma)
COLLECTION = None


def build_index(chunks):
    global COLLECTION
    import chromadb
    client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))
    try:
        client.delete_collection("cvs")                    # start clean on every reload
    except Exception:
        pass
    COLLECTION = client.create_collection("cvs", metadata={"hnsw:space": "cosine"})
    COLLECTION.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                   metadatas=[{"source": c["source"]} for c in chunks],
                   embeddings=embed([c["text"] for c in chunks]).tolist())
    return COLLECTION.count()


# ================================================================= LEVEL 3 · RETRIEVE AND PROMPT
# Prompt idea: "Implement retrieve: embed the query, query the Chroma collection for k results,
#  and return them as dicts with rank, source, text, score (1 - distance) and via='vector'.
#  Implement build_prompt: number the passages [1], [2]… with their source, then the question."
def retrieve(query, k=K):
    """Return the k best chunks: [{"rank": 1, "source": ..., "text": ..., "score": ..., "via": "vector"}, ...]"""
    raise NotImplementedError("LEVEL 3: implement retrieve() in challenge/pipeline.py")


def build_prompt(question, hits):
    """The text the LLM receives: the numbered passages, then the question."""
    raise NotImplementedError("LEVEL 3: implement build_prompt() in challenge/pipeline.py")


# ================================================================= LEVEL 4 · FEATURES (defaults)
def make_query(question, history):
    return question


def should_refuse(query, hits):
    return False


def build_graph(chunks):
    return nx.Graph()
