"""
Lesson 11 · Context management                            (checkpoint 7)

PROBLEM  By the fourth question the prompt holds the whole chat plus many
         passages. It gets slower, costs more, and the key sentence is buried.
IDEA     Summarize older turns, keep only the reranked top passages, and
         compress each passage to the sentences that matter.
WATCH    What was summarized or removed, the prompt size before/after, the
         time, and whether the answer still cites its sources.

Run:  python lessons/11_context_management.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, show
from ragkit.context import compress, count_tokens, summarize_history
from ragkit.llm import ANSWER_SYSTEM, build_prompt
from ragkit.pipelines import course_retriever

show.lesson("11", "Context management",
            "Long chats and many passages make prompts big, slow and noisy.",
            "Summarize old turns, keep the top passages, compress them to what matters.")

R = course_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm, R.reranker)

chat = [
    "User: Explain hybrid search and reranking.",
    "Assistant: Hybrid search runs BM25 and vector search in parallel and merges the lists with RRF. "
    "A reranker is a cross-encoder that re-scores the top candidates.",
    "User: What are the advantages of the above?",
    "Assistant: Hybrid search covers exact terms and paraphrases; reranking puts the best evidence first.",
    "User: Which one would find an exact policy ID?",
    "Assistant: The keyword half of hybrid search (BM25) matches exact IDs such as POL-204.",
]
question = "Compare hybrid search and reranking using the course documents."


def run(prompt_text: str, label: str):
    t0 = time.perf_counter()
    out = llm.chat(ANSWER_SYSTEM, prompt_text, 200)
    ms = (time.perf_counter() - t0) * 1000
    show.answer_box(out, f"{label} · {count_tokens(prompt_text)} tokens in · {ms:.0f} ms")
    return count_tokens(prompt_text), ms


show.step("A · Everything: full chat + 8 hybrid passages")
big_hits = R.hybrid(question, 8)
prompt_a = "Conversation so far:\n" + "\n".join(chat) + "\n\n" + build_prompt(question, [h.chunk.text for h in big_hits])
tok_a, ms_a = run(prompt_a, "Full prompt")

show.step("B · Managed: summary + last turn, reranked top 3, compressed passages")
summary, recent = summarize_history(chat, llm, keep_last=2)
show.info(f"   Summary of older turns: {summary}")
top = R.hybrid_rerank(question, 3)
compressed = [f"{h.chunk.section}: {compress(question, h.chunk.text, 2)}" for h in top]
for h, c in zip(top, compressed):
    show.info(f"   {h.chunk.label}: {len(h.chunk.text)} → {len(c) - len(h.chunk.section) - 2} characters")
removed = [h.chunk.label for h in big_hits if h.chunk.id not in {t.chunk.id for t in top}]
show.info(f"   Removed passages: {', '.join(removed)}")
prompt_b = f"Conversation summary: {summary}\n" + "\n".join(recent) + "\n\n" + build_prompt(question, compressed)
tok_b, ms_b = run(prompt_b, "Managed prompt")

show.table([{"prompt": "A · everything", "tokens": tok_a, "ms": round(ms_a)},
            {"prompt": "B · managed", "tokens": tok_b, "ms": round(ms_b)},
            {"prompt": "saving", "tokens": f"{100 * (1 - tok_b / tok_a):.0f}%", "ms": f"{ms_a - ms_b:.0f} ms"}],
           "Prompt size")
show.takeaway("Smaller prompts are cheaper and often clearer. But re-check the citations:\n"
              "if compression removed the sentence the answer needed, groundedness drops.")
