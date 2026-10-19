"""
Lesson 17 · RAG vs long context vs CAG: when do you need retrieval at all?

PROBLEM  Our course corpus is small. Maybe we could just paste all of it into
         the prompt (long context), or preload it once into a cache (CAG,
         cache-augmented generation), and skip retrieval entirely.
IDEA     Measure, don't guess: same questions, same model, three approaches.
         Then price them with YOUR provider's current rates.
WATCH    Tokens per question, time, answers, and the break-even table.
NOTE     The rates below are EXAMPLE NUMBERS, not real prices. Replace them with
         the price sheet for your provider on the day; prices change often.

Run:  python lessons/17_rag_vs_long_context_cag.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, load_course, show
from ragkit.context import count_tokens
from ragkit.llm import ANSWER_SYSTEM, build_prompt
from ragkit.pipelines import course_retriever

# Example rates per 1M tokens. REPLACE with real rates from your provider.
RATES = {"input": 1.00, "cached_input": 0.10, "cache_write": 1.25, "output": 4.00, "embedding": 0.02}
OUTPUT_TOKENS = 80          # typical answer length
CORPUS_UPDATES_PER_MONTH = 4

show.lesson("17", "RAG vs long context vs CAG",
            "For a small, stable corpus, retrieval may not be needed at all.",
            "Run the same questions three ways and price them with real rates.")

R = course_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)
corpus = "\n\n".join(f"# {d.title}\n{d.text}" for d in load_course())
corpus_tokens = count_tokens(corpus)
show.info(f"   Whole course corpus: {corpus_tokens} tokens")

questions = ["What is the capstone deadline?", "If I submit the capstone late, does that affect my team?"]
rows = []
for q in questions:
    show.step(q)
    hits = R.hybrid_rerank(q, 3)
    rag_prompt = build_prompt(q, [h.chunk.text for h in hits])
    lc_prompt = build_prompt(q, [corpus])
    for name, prompt in [("RAG (top 3)", rag_prompt), ("Long context", lc_prompt)]:
        t0 = time.perf_counter()
        out = llm.chat(ANSWER_SYSTEM, prompt, OUTPUT_TOKENS)
        ms = (time.perf_counter() - t0) * 1000
        show.answer_box(out, f"{name} · {count_tokens(prompt)} tokens · {ms:.0f} ms")
        rows.append({"question": q[:40], "approach": name, "prompt tokens": count_tokens(prompt), "ms": round(ms)})
show.table(rows, "Measured on this machine")
show.info("   CAG would send the same tokens as long context, but the corpus part is read from a cache\n"
          "   (prefix/KV cache) instead of being processed again, which is faster and usually billed lower.")

show.step("Cost per month (EXAMPLE RATES, replace them)")
rag_in = sum(r["prompt tokens"] for r in rows if r["approach"].startswith("RAG")) / len(questions)
q_tokens = 20


def per_month(n: int) -> dict:
    m = 1 / 1_000_000
    rag = n * ((rag_in * RATES["input"] + OUTPUT_TOKENS * RATES["output"] + q_tokens * RATES["embedding"]) * m) \
        + CORPUS_UPDATES_PER_MONTH * corpus_tokens * RATES["embedding"] * m
    lc = n * ((corpus_tokens + q_tokens) * RATES["input"] + OUTPUT_TOKENS * RATES["output"]) * m
    cag = CORPUS_UPDATES_PER_MONTH * corpus_tokens * RATES["cache_write"] * m \
        + n * (corpus_tokens * RATES["cached_input"] + q_tokens * RATES["input"] + OUTPUT_TOKENS * RATES["output"]) * m
    return {"questions/month": n, "RAG $": rag, "long context $": lc, "CAG $": cag}


show.table([per_month(n) for n in (100, 1_000, 10_000, 100_000)])
show.info("   Formulas: RAG = query embedding + retrieval + input(prompt + chunks) + output + indexing share\n"
          "             Long context = input(prompt + full corpus) + output\n"
          "             CAG = cache creation/refresh + cached input + output")
show.takeaway("RAG is not always cheaper or better. Small, stable corpus: long context or CAG can win.\n"
              "Large or changing corpus, permissions, citations at scale: RAG wins. Measure with real rates.")
