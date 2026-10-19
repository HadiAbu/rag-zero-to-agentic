"""
Lesson 07 · Reranking                                     (checkpoint 6)

PROBLEM  Hybrid search finds the right passages, but not always in the right
         order: for ERR-5007 it ranked ERR-5001 first, and for "journalists
         after a breach" a password policy came first.
IDEA     Retrieve a WIDER candidate set (10), then let a cross-encoder read the
         question and each candidate together and re-score them. Keep the top 3.
WATCH    Two separate operations: similarity search finds candidates, the
         reranker reorders them. Reranker scores (e.g. 6.2 or -10.1) are on a
         different scale than cosine or RRF scores, so compare ORDER, not numbers.
COST     One model pass per candidate. The lesson prints the added time.

Run:  python lessons/07_reranking.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import show
from ragkit.evaluate import load_golden, run_eval
from ragkit.pipelines import company_retriever, course_retriever
from ragkit.retrieval import rrf

show.lesson("07", "Reranking",
            "The right passage is in the candidates, but not at the top.",
            "Stage 1: hybrid search returns 10 candidates. Stage 2: a cross-encoder reorders them.")

K = company_retriever()
C = course_retriever(embedder=K.embedder, reranker=K.reranker)
show.setup_info(K.embedder, reranker=K.reranker)

cases = [
    (K, "Who can talk to journalists after a breach?", lambda c: "communications lead" in c.text),
    (K, "Can we retry ERR-5007?", lambda c: c.section == "ERR-5007"),
    (C, "Why would I combine keyword and vector search?", lambda c: "main advantage" in c.text),
]
rows = []
for R, q, expected in cases:
    show.step(q)
    candidates = rrf(R.vector(q, 10), R.keyword(q, 10), top=10)
    show.hits_table(candidates[:5], "Stage 1 · hybrid (top 5 of 10 candidates)", expected=expected)
    t0 = time.perf_counter()
    reranked = R.reranker.rerank(q, candidates, top=3)
    ms = (time.perf_counter() - t0) * 1000
    show.hits_table(reranked, f"Stage 2 · reranked top 3 ({ms:.0f} ms for 10 pairs)", expected=expected)
    before = next((h.rank for h in candidates if expected(h.chunk)), None)
    after = next((h.rank for h in reranked if expected(h.chunk)), None)
    rows.append({"question": q, "rank before": before or "miss", "rank after": after or "miss", "rerank ms": round(ms)})
show.table(rows, "Before vs after reranking")

show.step("Scoreboard: does reranking help on all test questions?")
board = []
for name, R, file in [("course", C, "course_questions.jsonl"), ("company", K, "company_questions.jsonl")]:
    golden = load_golden(file)
    for method in ["hybrid", "hybrid_rerank"]:
        res = run_eval(lambda item: getattr(R, method)(item["question"], 3), golden)
        board.append({"corpus": name, "method": method, "hit@3": res["hit_rate"], "MRR": res["mrr"], "avg ms": res["avg_ms"]})
show.table(board)
show.takeaway("Retrieval chooses candidates; reranking improves their order (higher MRR).\n"
              "It costs latency on every question, so check that better answers justify it.")
