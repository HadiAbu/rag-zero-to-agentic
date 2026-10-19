"""Prompt-size tools: count tokens, summarize old chat, compress passages."""
from __future__ import annotations

import re

from .llm import get_llm
from .retrieval import tokenize


def count_tokens(text: str) -> int:
    """Rough count: ~1.3 tokens per English word. Good enough to compare options."""
    return int(len(re.findall(r"\S+", text)) * 1.3) + 1


SUMMARY_SYSTEM = "Summarize this conversation in at most 2 sentences. Keep names, numbers and topics."


def summarize_history(turns: list[str], llm=None, keep_last: int = 1) -> tuple[str, list[str]]:
    """Return (summary of older turns, the most recent turns kept verbatim)."""
    old, recent = turns[:-keep_last] if keep_last else turns, turns[-keep_last:] if keep_last else []
    if not old:
        return "", recent
    llm = llm or get_llm()
    if getattr(llm, "can_reason", False):
        summary = llm.chat(SUMMARY_SYSTEM, "\n".join(old), 80)
    else:  # fallback: first sentence of each old turn
        summary = " ".join(re.split(r"(?<=[.!?])\s+", t)[0] for t in old)
    return summary, recent


def compress(question: str, text: str, max_sentences: int = 2) -> str:
    """Keep only the sentences that share words with the question (extractive compression)."""
    q = set(tokenize(question))
    sents = re.split(r"(?<=[.!?])\s+", text)
    scored = sorted(((len(q & set(tokenize(s))), i, s) for i, s in enumerate(sents)), reverse=True)
    keep = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(s for score, _, s in keep if score > 0) or sents[0]
