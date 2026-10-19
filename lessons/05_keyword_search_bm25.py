"""
Lesson 05 · Exact terms need keyword search               (checkpoint 4)

PROBLEM  Vector search matches meaning, so codes that look alike (ERR-5003,
         ERR-5007, ERR-5009 ...) blur together. Ask about ERR-5007 and you may
         get ERR-5003.
IDEA     BM25 keyword search scores exact word matches, weighted by how rare the
         word is. It nails IDs, codes and names. But it fails on paraphrases.
WATCH    The exact-code questions (BM25 wins) and the paraphrase questions
         (vector wins). Neither is enough alone.

Run:  python lessons/05_keyword_search_bm25.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import show
from ragkit.pipelines import company_retriever
from ragkit.retrieval import tokenize

show.lesson("05", "Keyword search (BM25)",
            "Vector search blurs IDs and error codes that look alike.",
            "Add BM25: exact tokens, weighted by rarity. Compare it with vector search.")

R = company_retriever()
show.setup_info(R.embedder)

cases = [
    ("Exact code", "Can we retry ERR-5007?", lambda c: c.section == "ERR-5007"),
    ("Exact code", "Should we retry ERR-5004?", lambda c: c.section == "ERR-5004"),
    ("Paraphrase", "Can I fly in a nicer seat on a long trip?", lambda c: c.doc_id == "travel-expenses" and "Business class" in c.text),
    ("Paraphrase", "Who can talk to journalists after a breach?", lambda c: "communications lead" in c.text),
]
rows = []
for kind, q, expected in cases:
    show.step(f"{kind}: {q}")
    v, b = R.vector(q, 3), R.keyword(q, 3)
    show.hits_table(v, "Vector (meaning)", expected=expected)
    show.hits_table(b, "BM25 (exact words)", expected=expected) if b else show.bad("BM25 found nothing: no shared words")
    rank = lambda hits: next((h.rank for h in hits if expected(h.chunk)), "miss")
    rows.append({"type": kind, "question": q, "vector": rank(v), "BM25": rank(b)})

show.table(rows, "Rank of the right passage")

show.step("Why BM25 found ERR-5007: the score per query word")
q = "Can we retry ERR-5007?"
for t in tokenize(q):
    show.info(f"   '{t}': idf = {R.bm25.idf.get(t, 0):.2f}  (higher = rarer = more points when it matches)")

show.takeaway("Keywords for exact terms, vectors for meaning. Run both → next lesson merges them.")
