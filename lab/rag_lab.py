"""
RAG LAB · build a RAG system one step at a time
================================================

HOW THIS FILE WORKS
  The pipeline is five small functions (make_query, retrieve, postprocess,
  build_context, should_refuse). STEP 1 is switched on. Every later STEP is a
  commented-out block that REPLACES one of those functions with a better one.

  1. Run it:                 python lab/rag_lab.py
  2. Read the scoreboard at the bottom of the output.
  3. Uncomment the next STEP block: select its lines and press Ctrl+/ (Cmd+/ on a Mac).
  4. Run again and compare. Then do the "YOUR TURN" task of that step.

  Steps 1-7 use the course documents and the 4 Build Day questions (one chat).
  Step 8 switches to the company documents, which have versions and permissions.

  Fast by default: only retrieval runs. Set GENERATE_ANSWERS = True to also
  get the LLM's answers (about 25 s per question on a laptop CPU).
"""
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import BM25Index, Reranker, VectorIndex, chunk_documents, get_embedder, get_llm, load_company, load_course
from ragkit.context import compress, count_tokens
from ragkit.llm import ANSWER_SYSTEM, build_prompt
from ragkit.query import rewrite_query
from ragkit.retrieval import Hit

# ---------------------------------------------------------------- settings
CORPUS = "course"          # "course" or "company" (STEP 8 switches it)
ROLE = "employee"          # who is asking (used from STEP 8)
K = 3                      # passages sent to the LLM
POOL = 10                  # candidates fetched before postprocess
GENERATE_ANSWERS = False   # True = also call the LLM for each answer (slow on CPU)
REFUSAL = "I don't know based on the provided documents."


# ---------------------------------------------------------------- STEP 1 · Naive RAG (always on)
# Embed the question, take the closest chunks by cosine similarity, answer from them.
def make_query(question, history):
    """What do we actually search for?"""
    return question


def retrieve(query, n):
    """Find candidate chunks."""
    return VEC.search(query, n, where=FILTER)


def postprocess(query, hits):
    """Choose which candidates go into the prompt."""
    return hits[:K]


def build_context(query, hits):
    """Turn the chosen chunks into the text the LLM reads."""
    return [h.chunk.with_context() for h in hits]


def should_refuse(query, hits):
    """Refuse before calling the LLM?"""
    return False


FILTER = None   # a function chunk -> bool that hides chunks (STEP 8)


## ======================================================================
## STEP 2 · Follow-up questions: add the chat history to the query
## You saw: "What are the advantages of the above?" has nothing to search for.
## ======================================================================
# def make_query(question, history):
#     return " ".join(history[-2:] + [question])
#
## YOUR TURN: change -2 to -1, then to -4. Which question gets worse, and why?


## ======================================================================
## STEP 3 · Query rewriting: one LLM call turns the follow-up into a full question
## You saw: raw history works, but it drags old words into the search.
## ======================================================================
# def make_query(question, history):
#     query, how = rewrite_query(history, question, llm())
#     print(f"      rewrite ({how}): {query}")
#     return query
#
## YOUR TURN: open ragkit/query.py and read REWRITE_SYSTEM. Add one sentence to the
## instruction and check whether the rewrites change.


## ======================================================================
## STEP 4 · Keyword search (BM25): exact words, rare words count more
## You saw (in lesson 05): vectors blur look-alike codes such as ERR-5007 and ERR-5003.
## ======================================================================
# def retrieve(query, n):
#     return BM25.search(query, n, where=FILTER)
#
## YOUR TURN: BM25 wins on Q3. Ask a paraphrase it cannot answer, for example
## "Which method understands synonyms?" (add it to QUESTIONS below).


## ======================================================================
## STEP 5 · Hybrid search: run both, merge the lists by rank (RRF)
## You saw: each retriever finds what the other misses.
## ======================================================================
# def retrieve(query, n):
#     vector_hits = VEC.search(query, 10, where=FILTER)
#     keyword_hits = BM25.search(query, 10, where=FILTER)
#     score, chunk_of = {}, {}
#     for hits in (vector_hits, keyword_hits):
#         for h in hits:                                   # RRF: 1 / (60 + rank) per list
#             score[h.chunk.id] = score.get(h.chunk.id, 0) + 1 / (60 + h.rank)
#             chunk_of[h.chunk.id] = h.chunk
#     best = sorted(score, key=score.get, reverse=True)[:n]
#     return [Hit(chunk_of[c], score[c], rank) for rank, c in enumerate(best, 1)]
#
## YOUR TURN: change 60 to 1 and to 1000. What happens to the order, and why?


## ======================================================================
## STEP 6 · Reranking: a cross-encoder reads question + passage together
## You saw: the right passage is found, but not always first.
## ======================================================================
# def postprocess(query, hits):
#     return reranker().rerank(query, hits, K)
#
## YOUR TURN: print the time this adds (time.perf_counter()). Is it worth it?


## ======================================================================
## STEP 7 · Context management: keep only the sentences that matter
## You saw: by Q4 the prompt is big; tokens cost money and time.
## ======================================================================
# def build_context(query, hits):
#     return [h.chunk.with_context().split("\n")[0] + "\n" + compress(query, h.chunk.text, 2)
#             for h in hits]
#
## YOUR TURN: set GENERATE_ANSWERS = True. Does the answer still cite its sources?


## ======================================================================
## STEP 8 · Company documents: old versions and permissions
## You saw (slides): a 2024 policy beats the 2026 one; a restricted codename leaks.
## Run once with only the first line uncommented, then uncomment the filter too.
## ======================================================================
# CORPUS = "company"
#
# def FILTER(chunk):
#     if chunk.meta.get("status") == "superseded":                    # old version
#         return False
#     if chunk.meta.get("access") == "restricted":                    # permissions
#         return ROLE in chunk.meta.get("allowed_roles", [])
#     return True
#
## YOUR TURN: set ROLE = "security". Which result changes? Is that correct?


## ======================================================================
## STEP 9 · Grounding: refuse when nothing relevant was found
## You saw: retrieval ALWAYS returns something, even for "What is the CEO's salary?".
## ======================================================================
# GATE = 0.40
#
# def should_refuse(query, hits):
#     best = VEC.search(query, 1, where=FILTER)
#     return not best or best[0].score < GATE
#
## YOUR TURN: try GATE between 0.30 and 0.45. Can ONE number refuse the CEO question
## and still answer the spread-out one? Print best[0].score for both. (Step 10 helps.)


## ======================================================================
## STEP 10 · Graph RAG: follow entities and relations, not just similarity
## You saw: "Which error codes must never be retried automatically?" has 4 answers
## in 4 passages. A top-3 search can never return all of them.
## ======================================================================
# import networkx as nx
#
# ID = re.compile(r"\b(?:ERR-\d{4}|POL-\d{3}|EXT-\d+|A\d{2})\b")
# RETRY_RULES = [(r"never retry|do not retry|never charge again", "never"),
#                (r"retry up to|retry once|safe to retry", "allowed")]
#
# def extract(chunk):
#     """Entities and relations in one chunk. Rules here; an LLM in production (YOUR TURN)."""
#     text = chunk.with_context()
#     entities = set(ID.findall(text))
#     relations = []
#     for code in entities:
#         if code.startswith("ERR-") and code in chunk.label:            # the chunk is ABOUT this code
#             for pattern, value in RETRY_RULES:
#                 if re.search(pattern, text, re.I):
#                     relations.append((code, "retry", value))
#                     break
#     return entities, relations
#
# def build_graph(chunks):
#     g = nx.Graph()
#     for c in chunks:
#         entities, relations = extract(c)
#         for e in entities:
#             g.add_edge(("chunk", c.id), ("entity", e), kind="mentions")
#         for a, rel, b in relations:
#             g.nodes[("entity", a)][rel] = b
#         for a in entities:                                               # co-mentioned = related
#             for b in entities - {a}:
#                 g.add_edge(("entity", a), ("entity", b), kind="related")
#     print(f"   graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
#     return g
#
# def graph_search(query):
#     """1) a relation question -> every entity with that relation; 2) named IDs -> their 1-hop neighbourhood."""
#     found = []
#     if re.search(r"error codes?", query, re.I) and re.search(r"never|not .*retr", query, re.I):
#         found = [n for n, d in GRAPH.nodes(data=True) if d.get("retry") == "never"]
#     for e in ID.findall(query):
#         if ("entity", e) in GRAPH:
#             found += [("entity", e)] + [n for n in GRAPH.neighbors(("entity", e)) if n[0] == "entity"]
#     chunk_ids = {c for n in found for c in GRAPH.neighbors(n) if c[0] == "chunk"}
#     return [CHUNKS[cid] for _, cid in chunk_ids if FILTER is None or FILTER(CHUNKS[cid])], found
#
# GRAPH_ON = True
# _hybrid = retrieve                                                       # keep STEP 5 as a fallback
#
# def retrieve(query, n):
#     chunks, entities = graph_search(query)
#     if chunks:
#         print(f"      graph path: {[e[1] for e in entities]} -> {len(chunks)} chunks")
#         return [Hit(c, 1.0, r) for r, c in enumerate(chunks, 1)]
#     return _hybrid(query, n)
#
# def postprocess(query, hits):
#     if hits and hits[0].score == 1.0 and len(hits) > 1:                 # graph answer: keep them all
#         return hits[:8]
#     return reranker().rerank(query, hits, K)
#
# _gate = should_refuse
#
# def should_refuse(query, hits):
#     if hits and hits[0].score == 1.0:                                  # the graph found the entities:
#         return False                                                   # trust structure over similarity
#     return _gate(query, hits)
#
## YOUR TURN: ask "Which policy covers project A17?" and print the graph path.
## STRETCH: replace extract() with an LLM call that returns (subject, relation, object).


## ======================================================================
## STEP 11 · MCP: your pipeline as a tool for Claude or Kiro
## Nothing to uncomment here. Run:   python lab/mcp_server.py   (it imports this file,
## so it serves exactly the steps you switched on). See lab/README.md to connect it.
## ======================================================================


# ---------------------------------------------------------------- the test questions
QUESTIONS = {
    "course": [   # one chat, asked in this order (the Build Day sequence)
        ("Q1", "Explain hybrid search and reranking.", {"docs": ["search-notes"], "contains": "Hybrid search runs"}),
        ("Q2", "What are the advantages of the above?", {"docs": ["search-notes"], "contains": "main advantage"}),
        ("Q3", "Which one would find an exact policy ID?", {"docs": ["search-notes"], "contains": "exact terms"}),
        ("Q4", "Can you compare them using the course documents?", {"docs": ["search-notes"], "contains": ""}),
    ],
    "company": [  # independent questions
        ("exact", "Can we retry ERR-5007?", {"docs": ["error-catalog"], "contains": "payout"}),
        ("paraphrase", "Can I fly in a nicer seat on a long trip?", {"docs": ["travel-expenses"], "contains": "Business class"}),
        ("version", "How many days of annual leave do I get?", {"docs": ["leave-policy-2026"], "contains": "25 days", "stale": "leave-policy-2024"}),
        ("restricted", "What is the codename of the incident war room?", {"restricted": True}),
        ("not in docs", "What is the CEO's salary?", {"refuse": True}),
        ("spread-out", "Which error codes must never be retried automatically?", {"all": ["ERR-4012", "ERR-5004", "ERR-5007", "ERR-5100"]}),
    ],
}


# ---------------------------------------------------------------- engine (no need to edit below)
_LLM, _RERANKER = None, None


def llm():
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM


def reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER


def ask(question, history):
    """One full RAG turn with whatever steps are switched on. Returns a dict for printing or tools."""
    t0 = time.perf_counter()
    query = make_query(question, history)
    hits = postprocess(query, retrieve(query, POOL))
    refused = should_refuse(query, hits)
    texts = [] if refused else build_context(query, hits)
    answer = REFUSAL if refused else None
    if GENERATE_ANSWERS and not refused:
        answer = llm().chat(ANSWER_SYSTEM, build_prompt(query, texts), 250)
    return {"query": query, "hits": [] if refused else hits, "texts": texts, "refused": refused,
            "answer": answer, "tokens": count_tokens(" ".join(texts)), "ms": (time.perf_counter() - t0) * 1000}


def judge(result, expect):
    """Score one result. Returns (ok, label)."""
    hits = result["hits"]
    if expect.get("refuse"):
        return (True, "refused ✓") if result["refused"] or not hits else (False, "answered from unrelated text ✗")
    if expect.get("restricted"):
        leaked = any(h.chunk.meta.get("access") == "restricted" and ROLE not in h.chunk.meta.get("allowed_roles", [])
                     for h in hits)
        return (False, "LEAKED restricted text ✗") if leaked else (True, "protected ✓")
    if expect.get("all"):
        text = " ".join(h.chunk.with_context() for h in hits)
        got = [c for c in expect["all"] if c in text]
        ok = len(got) == len(expect["all"])
        return ok, f"{len(got)}/{len(expect['all'])} found " + ("✓" if ok else "✗")
    stale = [h.rank for h in hits if expect.get("stale") and h.chunk.doc_id == expect["stale"]]
    if stale:
        return False, f"OUTDATED version in the prompt (#{stale[0]}) ✗"
    for h in hits:
        if h.chunk.doc_id in expect["docs"] and expect["contains"].lower() in h.chunk.text.lower():
            return True, f"#{h.rank} ✓"
    return False, "miss ✗"


def setup():
    """Load the documents and build the indexes for the current CORPUS (and the graph from STEP 10)."""
    global VEC, BM25, CHUNKS, GRAPH, DOCS
    DOCS = load_course() if CORPUS == "course" else load_company()
    chunks = chunk_documents(DOCS)
    CHUNKS = {c.id: c for c in chunks}
    VEC = VectorIndex(get_embedder()).add(chunks)
    BM25 = BM25Index(chunks)
    if globals().get("GRAPH_ON"):
        GRAPH = build_graph(chunks)
    return DOCS, chunks


def main():
    from ragkit import show
    docs, chunks = setup()
    show.info(f"Corpus: {CORPUS} ({len(docs)} documents, {len(chunks)} chunks) · role: {ROLE} · K={K}")
    rows, history = [], []
    for name, question, expect in QUESTIONS[CORPUS]:
        result = ask(question, history if CORPUS == "course" else [])
        ok, label = judge(result, expect)
        show.step(f"{name}: {question}")
        if result["query"] != question:
            show.info(f"   searched for: {result['query']}")
        for h in result["hits"]:
            show.info(f"   {h.rank}. {h.chunk.label}  ({h.score:.3f})")
        if result["refused"]:
            show.info("   (refused before calling the LLM)")
        if result["answer"]:
            show.answer_box(result["answer"])
        rows.append({"question": name, "result": label, "passage tokens": result["tokens"], "ms": round(result["ms"])})
        history.append(question)
    show.table(rows, "Scoreboard")
    passed = sum(1 for r in rows if "✓" in r["result"])
    show.info(f"{passed}/{len(rows)} correct · total passage tokens {sum(r['passage tokens'] for r in rows)}")


VEC = BM25 = CHUNKS = GRAPH = DOCS = None

if __name__ == "__main__":
    main()
