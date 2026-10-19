"""
Lesson 04 · Query rewriting                               (checkpoint 3)

PROBLEM  Pasting chat history into the query adds noise (lesson 03).
IDEA     Ask an LLM to rewrite the follow-up into a standalone question, e.g.
         "What are the advantages of the above?" →
         "What are the advantages of hybrid search and reranking?"
WATCH    The original question beside the rewritten query, then the results.
         With a tiny local model the rewrite can be poor; ragkit then falls back
         to a simple rule (replace 'the above'/'them'/'which one' with the topic).
TRY      RAG_LLM=ollama or HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct for better rewrites.

Run:  python lessons/04_query_rewriting.py
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, show
from ragkit.documents import DATA
from ragkit.evaluate import first_relevant_rank
from ragkit.pipelines import course_retriever
from ragkit.query import rewrite_query

show.lesson("04", "Query rewriting",
            "Follow-ups are not searchable, and raw chat history is noisy.",
            "One small LLM call turns the follow-up into a standalone search query.")

conv = json.loads((DATA / "golden" / "conversation.json").read_text())["turns"]
R = course_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)

history, rows = [], []
for turn in conv:
    q = turn["question"]
    t0 = time.perf_counter()
    rewritten, method = rewrite_query(history, q, llm)
    ms = (time.perf_counter() - t0) * 1000
    show.step(f"User: {q}")
    show.info(f"   Rewritten ({method}, {ms:.0f} ms): {rewritten}")
    before, after = R.vector(q, 3), R.vector(rewritten, 3)
    is_exp = lambda c, t=turn: c.doc_id in t["expected_docs"] and t["contains"].lower() in c.text.lower()
    show.hits_table(after, "Results for the rewritten query", expected=is_exp)
    rows.append({"original": q[:42], "rewritten": rewritten[:58],
                 "rank before": first_relevant_rank(before, turn) or "miss",
                 "rank after": first_relevant_rank(after, turn) or "miss", "added ms": round(ms)})
    history.append(q)

show.table(rows, "Original vs rewritten")
show.takeaway("Always show what was actually searched. Rewriting costs one small LLM call\n"
              "(record the time), and it makes follow-ups behave like first questions.")
