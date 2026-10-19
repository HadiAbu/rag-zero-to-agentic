"""
Lesson 10 · Access control: never retrieve what the user may not see

PROBLEM  The security incident plan is restricted to the security team. A
         normal employee asks about the war-room codename, retrieval finds the
         restricted chunk, and the model happily repeats it.
IDEA     Every chunk carries `access` and `allowed_roles`. Apply the user's
         permissions as a FILTER inside retrieval, before ranking and long
         before the LLM. Never rely on the prompt ("don't reveal secrets").
WATCH    Same question, three roles.

Run:  python lessons/10_access_control.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import answer, get_llm, show
from ragkit.pipelines import company_retriever

show.lesson("10", "Access control",
            "Restricted documents leak if retrieval ignores who is asking.",
            "Filter chunks by the user's role before ranking. The LLM never sees what the user can't.")

R = company_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)


def can_see(role: str):
    def check(chunk) -> bool:
        if chunk.meta.get("access", "internal") != "restricted":
            return True
        allowed = chunk.meta.get("allowed_roles", [])
        allowed = allowed if isinstance(allowed, list) else [allowed]
        return role in allowed
    return check


q = "What is the codename of the incident war room?"
restricted = lambda c: c.meta.get("access") == "restricted"

show.step(f"Employee asks: {q}  (NO access filter)")
leaked = R.hybrid(q, 3)
show.hits_table(leaked, "Retrieved", expected=restricted, mark="🔒")
show.answer_box(answer(q, leaked, llm), "Answer")
show.bad("🔒 = RESTRICTED chunk: it reached the prompt of someone without access")

for role in ["employee", "security"]:
    show.step(f"Role '{role}' asks the same question (WITH access filter)")
    hits = R.hybrid(q, 3, where=can_see(role))
    show.hits_table(hits, "Retrieved", expected=restricted, mark="🔒")
    show.answer_box(answer(q, hits, llm), f"Answer for {role}")

show.takeaway("Security belongs in retrieval, not in the prompt. Filter by role or tenant first,\n"
              "log who saw which sources, and test it with questions like these.")
