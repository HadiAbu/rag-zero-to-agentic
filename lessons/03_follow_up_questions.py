"""
Lesson 03 · Follow-up questions break retrieval          (checkpoint 2)

PROBLEM  In a chat, people ask "What are the advantages of the above?".
         The retriever only sees that sentence, and "the above" means nothing.
IDEA     Add the last few chat messages to the retrieval query.
WATCH    Question-only retrieval misses or guesses. With chat history it finds
         the topic again, but the extra text also adds noise.

Run:  python lessons/03_follow_up_questions.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import show
from ragkit.documents import DATA
from ragkit.evaluate import first_relevant_rank
from ragkit.pipelines import course_retriever

show.lesson("03", "Follow-up questions",
            "\"What are the advantages of the above?\" contains nothing to search for.",
            "Give the retriever the conversation too (here: the previous user questions).")

conv = json.loads((DATA / "golden" / "conversation.json").read_text())["turns"]
R = course_retriever()
show.setup_info(R.embedder)

history: list[str] = []
rows = []
for turn in conv:
    q = turn["question"]
    show.step(f"User: {q}")

    only = R.vector(q, 3)
    show.hits_table(only, "A · Question only", expected=lambda c, t=turn: c.doc_id in t["expected_docs"] and t["contains"].lower() in c.text.lower())

    with_ctx_query = " ".join(history[-2:] + [q])
    ctx = R.vector(with_ctx_query, 3)
    if history:
        show.info(f"   Query sent: {with_ctx_query!r}")
        show.hits_table(ctx, "B · Question + last 2 user messages", expected=lambda c, t=turn: c.doc_id in t["expected_docs"] and t["contains"].lower() in c.text.lower())

    rows.append({"question": q[:45], "question only": first_relevant_rank(only, turn) or "miss",
                 "with history": first_relevant_rank(ctx, turn) or "miss"})
    history.append(q)

show.table(rows, "Rank of the expected passage (top 3)")
show.takeaway("Question-only retrieval fails on follow-ups (it can get lucky on a keyword like 'advantages').\n"
              "Adding history recovers the topic, but the whole history is also noise.\n"
              "Next: rewrite the follow-up into one clean standalone question.")
