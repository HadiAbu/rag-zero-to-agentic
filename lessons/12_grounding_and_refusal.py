"""
Lesson 12 · Grounding, citations and refusal

PROBLEM  Retrieval always returns *something*, even for "Who is the hiring
         manager?". A model given weak passages may invent a fluent answer.
IDEA     Three guards:
         1. A relevance gate: if the best passage is too weak, don't call the LLM.
         2. A grounded prompt: answer only from sources, cite [n], else say "I don't know".
         3. A citation check: every sentence of the answer should overlap a cited source.
WATCH    The score calibration table: where would YOU put the threshold?
         Note that cross-encoder scores for good paraphrases can be low too,
         so thresholds must be calibrated on your own questions.

Run:  python lessons/12_grounding_and_refusal.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, show
from ragkit.llm import ANSWER_SYSTEM, build_prompt
from ragkit.pipelines import company_retriever
from ragkit.retrieval import tokenize

show.lesson("12", "Grounding, citations and refusal",
            "Retrieval never returns 'nothing', and models fill gaps with guesses.",
            "Gate on relevance, ground the prompt, and check every citation.")

R = company_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)

show.step("1 · Calibrate a relevance gate (top cosine score of vector search)")
answerable = ["How many days of annual leave do I get?", "Can I fly in a nicer seat on a long trip?",
              "Who can talk to journalists after a breach?", "I lost my laptop, what now?"]
unanswerable = ["What is the CEO's salary?", "Do we have a dress code?", "What is the parking policy?",
                "Who is the hiring manager?"]
rows = []
for q in answerable + unanswerable:
    rows.append({"question": q, "answerable": q in answerable, "top cosine": R.vector(q, 1)[0].score})
show.table(sorted(rows, key=lambda r: -r["top cosine"]), "Top similarity: answerable vs not")
THRESHOLD = 0.40
show.info(f"   This lesson uses THRESHOLD = {THRESHOLD}. Change it and see who gets refused.")

LOOSE_SYSTEM = "You are a helpful assistant. Answer the question."


def citation_check(answer_text: str, sources: list[str]) -> list[str]:
    """Return sentences that share almost no words with the sources (likely unsupported)."""
    src = set(tokenize(" ".join(sources)))
    flagged = []
    for s in re.split(r"(?<=[.!?])\s+", answer_text):
        words = set(tokenize(s))
        if "don't know" in s.lower():
            continue
        if len(words) >= 3 and len(words & src) / len(words) < 0.4:
            flagged.append(s)
    return flagged


for q in ["What is the CEO's salary?", "Do we have a dress code?", "How many sick days do I have?"]:
    show.step(f"Question: {q}")
    hits = R.hybrid(q, 3, where=lambda c: c.meta.get("status") == "current")
    texts = [h.chunk.text for h in hits]
    top_cos = R.vector(q, 1)[0].score
    loose = llm.chat(LOOSE_SYSTEM, build_prompt(q, texts), 120)
    show.answer_box(loose, "No guards: loose prompt")
    if top_cos < THRESHOLD:
        show.good(f"Gate: top similarity {top_cos:.2f} < {THRESHOLD} → refuse without calling the LLM")
        show.answer_box("I don't know based on the provided documents.", "With guards")
        continue
    grounded = llm.chat(ANSWER_SYSTEM, build_prompt(q, texts), 120)
    show.answer_box(grounded, "With guards: grounded prompt")
    flagged = citation_check(grounded, texts)
    if flagged:
        show.bad("Citation check flagged: " + " | ".join(flagged))
    else:
        show.good("Citation check: every sentence overlaps a source")

show.takeaway("Measure refusals too: did the system say 'I don't know' when the documents were silent?\n"
              "A fluent answer is not a correct answer. Check what it is grounded in.")
