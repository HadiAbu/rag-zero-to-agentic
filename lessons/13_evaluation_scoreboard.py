"""
Lesson 13 · Evaluation: the scoreboard

PROBLEM  Every lesson so far "looked better". Looking is not measuring.
IDEA     A golden set: questions + the passage that should answer each. Run
         every pipeline on it and record, per checkpoint:
           Retrieval  - did the expected source appear in the top 3?
           Ranking    - how high did the best source appear (MRR = average of 1/rank)?
           Cost/speed - prompt tokens and milliseconds added
         Answer quality and refusals: see lesson 12.
WATCH    Part A: methods on both test sets. Part B: the Build Day conversation,
         checkpoint by checkpoint, exactly like the slides' scoreboard.

Run:  python lessons/13_evaluation_scoreboard.py
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, show
from ragkit.context import compress, count_tokens
from ragkit.documents import DATA
from ragkit.evaluate import first_relevant_rank, load_golden, run_eval
from ragkit.pipelines import company_retriever, course_retriever
from ragkit.query import rewrite_query

show.lesson("13", "Evaluation scoreboard",
            "Changes 'look better', but we haven't measured anything.",
            "Same questions, same documents, one row per configuration. Change one thing at a time.")

C = course_retriever()
K = company_retriever(embedder=C.embedder, reranker=C.reranker)
show.setup_info(C.embedder, reranker=C.reranker)

# ---------------------------------------------------------------- Part A
show.step("Part A · Retrieval methods on both test sets (top 3)")
rows = []
for corpus, R, file in [("course", C, "course_questions.jsonl"), ("company", K, "company_questions.jsonl")]:
    golden = load_golden(file)
    for method in ["vector", "keyword", "hybrid", "hybrid_rerank"]:
        res = run_eval(lambda item: getattr(R, method)(item["question"], 3), golden)
        rows.append({"corpus": corpus, "method": method, "hit@3": res["hit_rate"], "MRR": res["mrr"],
                     "avg ms": res["avg_ms"], "misses": ", ".join(r["id"] for r in res["rows"] if not r["hit"]) or "-"})
show.table(rows)

# ---------------------------------------------------------------- Part B
show.step("Part B · Build Day conversation, checkpoint by checkpoint")
turns = json.loads((DATA / "golden" / "conversation.json").read_text())["turns"]
llm = get_llm()
questions = [t["question"] for t in turns]
rewrites, rewrite_ms = [], []
for i, q in enumerate(questions):
    t0 = time.perf_counter()
    rewrites.append(rewrite_query(questions[:i], q, llm)[0])
    rewrite_ms.append((time.perf_counter() - t0) * 1000)

checkpoints = {
    "1 basic (vector)": lambda i: C.vector(questions[i], 3),
    "2 + chat context": lambda i: C.vector(" ".join(questions[max(0, i - 2):i + 1]), 3),
    "3 + rewriting": lambda i: C.vector(rewrites[i], 3),
    "4 keyword (BM25)": lambda i: C.keyword(rewrites[i], 3),
    "5 hybrid (RRF)": lambda i: C.hybrid(rewrites[i], 3),
    "6 + reranking": lambda i: C.hybrid_rerank(rewrites[i], 3),
    "7 + compression": lambda i: C.hybrid_rerank(rewrites[i], 3),
}
board = []
for name, run in checkpoints.items():
    row = {"checkpoint": name}
    tokens = 0
    t0 = time.perf_counter()
    for i, turn in enumerate(turns):
        hits = run(i)
        r = first_relevant_rank(hits, turn)
        row[f"Q{i + 1}"] = f"#{r}" if r else "miss"
        texts = [compress(rewrites[i], h.chunk.text, 2) if name.startswith("7") else h.chunk.text for h in hits]
        tokens += count_tokens(" ".join(texts))
    ms = (time.perf_counter() - t0) * 1000 + (sum(rewrite_ms) if name[0] in "34567" else 0)
    row["passage tokens"] = tokens
    row["ms (4 Qs)"] = round(ms)
    board.append(row)
show.table(board, "Rank of the expected passage per question (miss = not in top 3)")
show.info("Rewrites used: " + " | ".join(rewrites))
show.takeaway("Keep this table. Every new idea (chunk size, model, reranker, prompt) gets a new row,\n"
              "and only changes that move the numbers, at a cost you accept, stay.")
