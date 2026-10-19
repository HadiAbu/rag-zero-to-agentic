"""Language-model backends. All free options come first.

RAG_LLM=auto (default)  Ollama if it is running, else a small local Hugging Face model,
                        else the extractive fallback.
RAG_LLM=ollama          Ollama (https://ollama.com), model from OLLAMA_MODEL (default llama3.2:3b)
RAG_LLM=hf              Qwen/Qwen2.5-1.5B-Instruct via transformers (~3 GB download, CPU is fine)
                        Faster/smaller: HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct (~1 GB, weaker reasoning)
RAG_LLM=openai          Any OpenAI-compatible server: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
                        (works with LM Studio, vLLM, llama.cpp server, or a hosted API)
RAG_LLM=extractive      No model: copies the best-matching sentences. Cannot rewrite or reason.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .retrieval import tokenize


class ExtractiveLLM:
    """Not a language model. Picks the context sentences that overlap most with the
    question. Good enough to show retrieval working; it cannot rewrite, plan or summarize."""

    name = "extractive (no model)"
    can_reason = False

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        if "Context:" not in user:
            return ""
        context, _, question = user.partition("Question:")
        q = set(tokenize(question))
        best = []
        for m in re.finditer(r"\[(\d+)\]([^\[]+)", context):
            n, text = m.group(1), m.group(2)
            for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    best.append((overlap, -int(n), sent.strip(), n))
        if not best:
            return "I don't know based on the provided documents."
        best.sort(reverse=True)
        return " ".join(f"{s} [{n}]" for _, _, s, n in best[:2])


class OllamaLLM:
    can_reason = True

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{self.model}"

    @staticmethod
    def available(host: str | None = None) -> bool:
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            urllib.request.urlopen(host + "/api/tags", timeout=0.5)
            return True
        except Exception:  # noqa: BLE001
            return False

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()


_HF: dict = {}


class HFLocalLLM:
    can_reason = True

    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.name = f"hf:{self.model_name}"
        if self.model_name not in _HF:
            from transformers import pipeline

            _HF[self.model_name] = pipeline("text-generation", model=self.model_name)
        self.pipe = _HF[self.model_name]

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self.pipe(msgs, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


class OpenAICompatLLM:
    can_reason = True

    def __init__(self):
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.getenv("OPENAI_API_KEY", "not-needed")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai-compatible:{self.model}"

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()


_LLM = None


def get_llm():
    """Return one shared LLM for the whole run (models are loaded once)."""
    global _LLM
    if _LLM is not None:
        return _LLM
    choice = os.getenv("RAG_LLM", "auto").lower()
    if choice == "extractive":
        _LLM = ExtractiveLLM()
    elif choice == "ollama":
        _LLM = OllamaLLM()
    elif choice == "openai":
        _LLM = OpenAICompatLLM()
    elif choice == "hf":
        _LLM = HFLocalLLM()
    else:  # auto
        if OllamaLLM.available():
            _LLM = OllamaLLM()
        else:
            try:
                _LLM = HFLocalLLM()
            except Exception:  # noqa: BLE001
                _LLM = ExtractiveLLM()
    return _LLM


# ---------------------------------------------------------------- prompts
ANSWER_SYSTEM = ("You answer questions using ONLY the numbered sources in the context. "
                 "Cite every fact like [1]. If the sources do not contain the answer, reply exactly: "
                 "\"I don't know based on the provided documents.\" Be brief.")


def build_prompt(question: str, texts: list[str]) -> str:
    ctx = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def answer(question: str, hits, llm=None, max_tokens: int = 200) -> str:
    llm = llm or get_llm()
    texts = [h.chunk.with_context() if hasattr(h, "chunk") else str(h) for h in hits]
    return llm.chat(ANSWER_SYSTEM, build_prompt(question, texts), max_tokens)
