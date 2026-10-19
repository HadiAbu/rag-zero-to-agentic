"""
Lesson 15 · Company-grade ("corporate") RAG: everything together

PROBLEM  Real systems need all the fixes at once, plus things a demo ignores:
         permissions, outdated documents, follow-ups, citations people can
         check, refusals, logs, and cost.
IDEA     One pipeline with explicit stages:
           rewrite → filter (role + current) → hybrid → rerank → relevance gate
           → compress → grounded answer with citations → log + cache
WATCH    The per-stage trace (ms) for each question and the citations
         (document, version, section). Logs go to logs/rag_log.jsonl.

Run:  python lessons/15_production_rag.py
"""
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import chunk_documents, get_llm, load_company, show
from ragkit.context import compress, count_tokens
from ragkit.llm import ANSWER_SYSTEM, build_prompt
from ragkit.pipelines import Retriever
from ragkit.query import rewrite_query

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class User:
    name: str
    role: str
    history: list = field(default_factory=list)


class CompanyRAG:
    def __init__(self, gate: float = 0.40):
        self.retriever = Retriever(chunk_documents(load_company()))
        self.llm = get_llm()
        self.gate = gate
        self.cache: dict = {}
        self.log_path = ROOT / "logs" / "rag_log.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

    @staticmethod
    def allowed(role: str):
        def check(c) -> bool:
            if c.meta.get("status", "current") != "current":
                return False                     # never answer from superseded versions
            if c.meta.get("access") == "restricted":
                roles = c.meta.get("allowed_roles", [])
                return role in (roles if isinstance(roles, list) else [roles])
            return True
        return check

    def ask(self, user: User, question: str) -> dict:
        trace, t = {}, time.perf_counter()

        def mark(stage):
            nonlocal t
            now = time.perf_counter()
            trace[stage] = round((now - t) * 1000)
            t = now

        query, method = rewrite_query(user.history, question, self.llm)
        mark("rewrite")
        key = (user.role, query.lower())
        if key in self.cache:
            out = dict(self.cache[key], cached=True, trace={"cache": 0})
            user.history.append(question)
            return out
        where = self.allowed(user.role)
        top_cos = (self.retriever.vector(query, 1, where) or [None])[0]
        candidates = self.retriever.hybrid(query, 10, where)
        mark("hybrid")
        top = self.retriever.reranker.rerank(query, candidates, 3) if candidates else []
        mark("rerank")
        if not top or top_cos is None or top_cos.score < self.gate:
            result = {"answer": "I don't know based on the documents you can access.", "sources": [], "refused": True}
        else:
            passages = [f"{h.chunk.meta.get('title')} › {h.chunk.section}: {compress(query, h.chunk.text, 3)}" for h in top]
            mark("compress")
            prompt = build_prompt(query, passages)
            text = self.llm.chat(ANSWER_SYSTEM, prompt, 200)
            mark("generate")
            result = {"answer": text, "refused": False, "prompt_tokens": count_tokens(prompt),
                      "sources": [f"[{i}] {h.chunk.meta.get('title')} v{h.chunk.meta.get('version')} › {h.chunk.section}"
                                  for i, h in enumerate(top, 1)]}
        result.update(query=query, rewrite=method, trace=trace, cached=False)
        self.cache[key] = result
        user.history.append(question)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user.name, "role": user.role, "question": question, **result}) + "\n")
        return result


show.lesson("15", "Company-grade RAG",
            "A real assistant needs permissions, fresh documents, follow-ups, citations, refusals and logs.",
            "One explicit pipeline: rewrite → filter → hybrid → rerank → gate → compress → answer → log.")

rag = CompanyRAG()
show.setup_info(rag.retriever.embedder, rag.llm, rag.retriever.reranker)
dana = User("Dana", "employee")
sam = User("Sam", "engineer")
lee = User("Lee", "security")
script = [
    (dana, "How many days of annual leave do I get?"),
    (dana, "And how many of them can I carry over?"),
    (sam, "Can we retry ERR-5007?"),
    (dana, "What is the codename of the incident war room?"),
    (lee, "What is the codename of the incident war room?"),
    (dana, "What is the CEO's salary?"),
    (dana, "How many days of annual leave do I get?"),
]
for user, q in script:
    show.step(f"{user.name} ({user.role}): {q}")
    r = rag.ask(user, q)
    if r["query"] != q:
        show.info(f"   searched for: {r['query']}  ({r['rewrite']})")
    show.answer_box(r["answer"] + ("\n\n" + "\n".join(r["sources"]) if r["sources"] else ""),
                    "Answer" + (" (from cache)" if r["cached"] else "") + (" · refused" if r["refused"] else ""))
    show.info("   trace ms: " + ", ".join(f"{k}={v}" for k, v in r["trace"].items()))

show.info(f"Log written to {rag.log_path.relative_to(ROOT)}")
show.takeaway("Production RAG is retrieval engineering: filters, freshness, ranking, refusals,\n"
              "citations, logs and caching. The LLM call is one stage of eight.")
