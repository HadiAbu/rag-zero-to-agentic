"""
Lesson 01 · Naive RAG                                   (Build Day checkpoint 1)

PROBLEM  A language model only knows its training data. It has never seen our
         course documents, so it cannot answer "When is the capstone due?".
IDEA     Retrieval-Augmented Generation: cut the documents into chunks, embed
         them, find the chunks closest to the question, and give those chunks
         to the model together with the question.
WATCH    The retrieved chunks and their similarity scores. Naive RAG works for
         easy questions, but fixed-size chunks cut sentences in half.
TRY      Change CHUNK_SIZE to 80 or 600 and run again.

Run:  python lessons/01_naive_rag.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import VectorIndex, answer, chunk_documents, get_embedder, get_llm, load_course
from ragkit import show

CHUNK_SIZE = 200  # characters, cut blindly (that is what makes it "naive")

show.lesson("01", "Naive RAG",
            "The model has never seen our course documents.",
            "Chunk → embed → retrieve the top 3 by vector similarity → answer from them.")

# 1. Load documents ----------------------------------------------------------
docs = load_course()
show.step(f"1. Load {len(docs)} course documents")
for d in docs:
    show.info(f"   {d.meta['source']:32s} {len(d.text):5d} characters")

# 2. Chunk: fixed size, no overlap ------------------------------------------
chunks = chunk_documents(docs, strategy="fixed", size=CHUNK_SIZE)
show.step(f"2. Cut them into {len(chunks)} fixed-size chunks of {CHUNK_SIZE} characters")
show.info("   Example chunk: " + repr(chunks[3].text[:CHUNK_SIZE]))

# 3. Embed and index ---------------------------------------------------------
embedder = get_embedder()
index = VectorIndex(embedder, contextual=False).add(chunks)
show.step(f"3. Embed every chunk into a vector ({index.vectors.shape[1]} numbers each)")
show.info(f"   First 6 numbers of chunk 0: {[round(float(x), 3) for x in index.vectors[0][:6]]}")

# 4. Ask ---------------------------------------------------------------------
llm = get_llm()
show.setup_info(embedder, llm)
questions = [
    "What is the capstone deadline?",            # easy: same words as the document
    "How do I ask for more time on the final project?",  # needs the extensions section
    "Who is the hiring manager?",                # not in the documents at all
]
for q in questions:
    show.step(f"4. Question: {q}")
    hits = index.search(q, k=3)
    show.hits_table(hits, "Top 3 chunks by cosine similarity")
    show.answer_box(answer(q, hits, llm))

show.takeaway(
    "Naive RAG already answers simple questions with sources.\n"
    "But look at the chunks: they start and stop in the middle of sentences, and the\n"
    "retriever only sees the question, nothing else. The next lessons fix one problem at a time.")
