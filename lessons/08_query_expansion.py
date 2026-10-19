"""
Lesson 08 · Query expansion: multi-query and HyDE         (stretch)

PROBLEM  Short or oddly-worded questions don't look like the passages that
         answer them ("Why would I combine keyword and vector search?").
IDEA 1   Multi-query: an LLM writes several phrasings, we search each and merge
         the lists with RRF ("RAG-Fusion").
IDEA 2   HyDE: an LLM writes a hypothetical answer; we search with that answer,
         because answer-shaped text sits closer to real answers. The draft may
         be wrong. It is only used for searching, never shown to the user.
WATCH    The generated text, the merged results, and the extra LLM time.
         More queries also bring more irrelevant results → rerank afterwards.

Run:  python lessons/08_query_expansion.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, rrf, show
from ragkit.pipelines import course_retriever
from ragkit.query import hyde, multi_query

show.lesson("08", "Query expansion",
            "The question's wording doesn't match the wording of the answer.",
            "Search with several phrasings (multi-query) or with a drafted answer (HyDE).")

R = course_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)
expected = lambda c: "main advantage" in c.text or "Extensions must be requested" in c.text

for q in ["Why would I combine keyword and vector search?", "How do I ask for more time on the final project?"]:
    show.step(f"Question: {q}")
    show.hits_table(R.vector(q, 3), "Plain vector search", expected=expected)

    t0 = time.perf_counter()
    queries = multi_query(q, llm)
    ms = (time.perf_counter() - t0) * 1000
    show.info(f"   Multi-query ({ms:.0f} ms):")
    for v in queries:
        show.info(f"     • {v}")
    merged = rrf(*[R.vector(v, 5) for v in queries], top=3)
    show.hits_table(merged, "Multi-query, merged with RRF", expected=expected)

    t0 = time.perf_counter()
    draft = hyde(q, llm)
    ms = (time.perf_counter() - t0) * 1000
    show.info(f"   HyDE draft ({ms:.0f} ms): {draft[:220]}")
    show.hits_table(R.vector(draft, 3), "HyDE: searched with the drafted answer", expected=expected)

show.takeaway("Expansion gives retrieval more ways in, at the cost of extra LLM calls and noisier\n"
              "candidate lists. Use it for vague questions, and rerank the merged results.")
