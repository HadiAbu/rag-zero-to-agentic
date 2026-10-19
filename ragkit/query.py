"""Query transformations: rewriting follow-ups, multi-query expansion and HyDE."""
from __future__ import annotations

import re

from .llm import get_llm
from .retrieval import tokenize

REFERRING = re.compile(r"\b(the above|them|it|this|that|these|those|which one)\b", re.I)
LEADING = re.compile(r"^(please\s+)?(explain|describe|what is|what are|tell me about|define|summarize)\s+", re.I)


def topic_of(question: str) -> str:
    """'Explain hybrid search and reranking.' -> 'hybrid search and reranking'"""
    return LEADING.sub("", question.strip()).rstrip(" ?.!")


def is_follow_up(question: str) -> bool:
    return bool(REFERRING.search(question))


def rule_rewrite(history: list[str], question: str) -> str:
    """No-model fallback: replace referring words with the topic of the last
    standalone question in the history."""
    topic, clean = "", False
    for past in reversed(history):
        if not is_follow_up(past):
            topic, clean = topic_of(past), bool(LEADING.match(past.strip()))
            break
    if not topic or not is_follow_up(question):
        return question
    if not clean:  # the previous question was not "Explain X": append its key words instead
        words = [w for w in tokenize(topic) if w not in {"many", "much", "get", "need", "know"}]
        return f"{question.rstrip('?')} ({' '.join(words)})?"

    def sub(m):
        word = m.group(0).lower()
        return f"which of {topic}" if word == "which one" else topic

    return REFERRING.sub(sub, question, count=1)


REWRITE_SYSTEM = (
    "You rewrite the user's LAST question so it can be understood without the conversation. "
    "Replace words like 'it', 'them', 'the above' or 'which one' with what they refer to. "
    "Keep every other word of the question. Return only the rewritten question.")

REWRITE_EXAMPLE = (
    "Conversation:\nUser: Tell me about the leave policy.\n"
    "Last question: How many days does it give?\n"
    "Rewritten: How many days does the leave policy give?\n\n")


def rewrite_query(history: list[str], question: str, llm=None) -> tuple[str, str]:
    """Return (standalone query, method used)."""
    if not history or not is_follow_up(question):
        return question, "unchanged"
    llm = llm or get_llm()
    if getattr(llm, "can_reason", False):
        convo = "\n".join(f"User: {h}" for h in history[-3:])
        out = llm.chat(REWRITE_SYSTEM, REWRITE_EXAMPLE + f"Conversation:\n{convo}\nLast question: {question}\nRewritten:", 60)
        out = out.strip().strip('"').split("\n")[0].replace("Rewritten:", "").strip()
        # sanity checks: long enough, no vague reference left, and it kept the topic words
        topic_words = set(tokenize(rule_rewrite(history, question))) - set(tokenize(question))
        if len(out.split()) >= 4 and not REFERRING.search(out) and topic_words <= set(tokenize(out)):
            return out, f"LLM ({llm.name})"
    return rule_rewrite(history, question), "rule-based fallback"


MULTI_SYSTEM = ("Write 3 different search queries that would help answer the user's question. "
                "Use different words and angles. One query per line, no numbering, nothing else.")


def multi_query(question: str, llm=None, n: int = 3) -> list[str]:
    llm = llm or get_llm()
    variants = []
    if getattr(llm, "can_reason", False):
        out = llm.chat(MULTI_SYSTEM, question, 120)
        variants = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip() for line in out.splitlines()]
        variants = [v for v in variants if len(v.split()) >= 3][:n]
    if not variants:  # fallback: keyword-only and topic-only variants
        topic = topic_of(question)
        variants = [" ".join(w for w in re.findall(r"\w[\w-]*", question) if len(w) > 3), topic]
    return [question] + [v for v in variants if v.lower() != question.lower()]


HYDE_SYSTEM = ("Write a short passage (2-3 sentences) that would answer the question, "
               "as if it came from a course handbook. Do not say you are unsure.")


def hyde(question: str, llm=None) -> str:
    """Hypothetical Document Embeddings: search with a made-up answer instead of the question."""
    llm = llm or get_llm()
    if getattr(llm, "can_reason", False):
        return llm.chat(HYDE_SYSTEM, question, 120)
    return question
