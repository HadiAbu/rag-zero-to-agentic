"""
Lesson 06 · Hybrid search with reciprocal rank fusion     (checkpoint 5)

PROBLEM  BM25 and vector search each fail on different questions (lesson 05),
         and their scores (9.1 vs 0.62) are not comparable, so we can't just add them.
IDEA     Reciprocal rank fusion: score(d) = Σ 1 / (60 + rank of d in each list).
         It uses ranks only, so the scales don't matter.
WATCH    The fused list keeps the exact match AND the paraphrase match.
         Then the scoreboard over all company test questions.

Run:  python lessons/06_hybrid_search_rrf.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import rrf, show
from ragkit.evaluate import load_golden, run_eval
from ragkit.pipelines import company_retriever

show.lesson("06", "Hybrid search + RRF",
            "Each retriever misses what the other one finds; their scores live on different scales.",
            "Merge the two ranked lists by rank: score = Σ 1/(60 + rank).")

R = company_retriever()
show.setup_info(R.embedder)

q = "Can we retry ERR-5007?"
show.step(f"Worked example: {q}")
v, b = R.vector(q, 5), R.keyword(q, 5)
fused = rrf(v, b, top=5)
show.hits_table(v, "Vector list", expected=lambda c: c.section == "ERR-5007")
show.hits_table(b, "BM25 list", expected=lambda c: c.section == "ERR-5007")
rows = []
for h in fused:
    d = h.details
    parts = [f"1/(60+{d[k]})" for k in ("list1_rank", "list2_rank") if k in d]
    rows.append({"rank": h.rank, "chunk": h.chunk.label, "vector rank": d.get("list1_rank", "-"),
                 "BM25 rank": d.get("list2_rank", "-"), "RRF": " + ".join(parts), "score": h.score})
show.table(rows, "Fused list (RRF, k = 60)")

show.step("Scoreboard on all company test questions (top 3)")
golden = load_golden("company_questions.jsonl")
board = []
for name in ["vector", "keyword", "hybrid"]:
    res = run_eval(lambda item: getattr(R, name)(item["question"], 3), golden)
    misses = [r["id"] for r in res["rows"] if not r["hit"]]
    board.append({"method": name, "hit@3": res["hit_rate"], "MRR": res["mrr"], "misses": ", ".join(misses) or "-"})
show.table(board)
show.takeaway("Hybrid covers both exact matches and paraphrases. RRF needs no score tuning.\n"
              "But the order inside the top 3 is still rough → reranking.")
