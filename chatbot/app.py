"""
CV CHATBOT · the starter (naive RAG)          python chatbot/app.py   →  http://127.0.0.1:7862

Put the CVs in chatbot/cvs/ (PDF, DOCX, TXT or MD). That folder is ignored by git: real CVs
never go to GitHub.

This file is deliberately simple. It is NAIVE RAG: fixed-size chunks, vector search, top-K,
one prompt. Your job today is to make it better, one step at a time, with Claude Code or Kiro,
and to PROVE every step is better in the Verify tab.

  Tab "Chat"    ask questions; the right side shows what was searched, what came back,
                the exact prompt sent to the LLM, and the answer with its timing.
  Tab "Verify"  upload a questions JSON (format: chatbot/questions_example.json), run them all,
                and get Hit@K, Recall@K, MRR, refusals and a row per question you can open.

Every place you can improve is marked  # STEP n · ...  (see chatbot/README.md for the menu).
"""
import json
import os
import pathlib
import re
import sys
import time

import gradio as gr
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragkit import get_embedder, get_llm      # the same free local models as the lab

CV_DIR = pathlib.Path(os.getenv("CV_DIR", pathlib.Path(__file__).parent / "cvs"))
K = 3                     # how many chunks go into the prompt
CHUNK_SIZE = 800          # characters per chunk
OVERLAP = 100             # characters shared by neighbouring chunks
REFUSAL = "I don't know based on the CVs."

SYSTEM_PROMPT = ("You are a recruiting assistant. Answer ONLY from the numbered CV passages. "
                 "Cite every fact like [1]. If the passages do not contain the answer, reply exactly: "
                 f"\"{REFUSAL}\" Be brief.")


# ================================================================= 1 · LOAD
def read_file(path: pathlib.Path) -> str:
    """Turn one CV file into plain text."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if path.suffix.lower() == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_cvs():
    files = sorted(p for p in CV_DIR.rglob("*") if p.suffix.lower() in (".pdf", ".docx", ".txt", ".md"))
    return [{"source": p.name, "text": read_file(p)} for p in files]


# ================================================================= 2 · CHUNK
# STEP 2 · Better chunks: split by section (Experience, Skills, Education...), and put the
#          candidate's name in front of every chunk. Right now a chunk from page 2 does not
#          even say whose CV it is.
def chunk(doc):
    text, out, start = re.sub(r"\s+", " ", doc["text"]).strip(), [], 0
    while start < len(text):
        out.append({"source": doc["source"], "text": text[start:start + CHUNK_SIZE]})
        start += CHUNK_SIZE - OVERLAP
    return out


# ================================================================= 3 · EMBED AND INDEX
def build_index():
    global CHUNKS, VECTORS
    docs = load_cvs()
    CHUNKS = [c for d in docs for c in chunk(d)]
    VECTORS = EMBEDDER.encode([c["text"] for c in CHUNKS])
    VECTORS = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)
    return f"Loaded {len(docs)} CVs from {CV_DIR} → {len(CHUNKS)} chunks"


# ================================================================= 4 · RETRIEVE
# STEP 1 · Follow-ups: "what about his education?" needs the chat history (rewrite the question).
# STEP 3 · Hybrid search: add BM25 for exact words (names, tools, certificates) + merge with RRF.
# STEP 4 · Rerank the top 20 with a cross-encoder (ragkit.Reranker) and keep K.
# STEP 5 · Filters: read "in Haifa", "5+ years" from the question and filter before searching.
def make_query(question, history):
    return question


def retrieve(query):
    q = EMBEDDER.encode([query])[0]
    q = q / np.linalg.norm(q)
    scores = VECTORS @ q
    best = np.argsort(-scores)[:K]
    return [{"rank": r, "score": float(scores[i]), **CHUNKS[i]} for r, i in enumerate(best, 1)]


# ================================================================= 5 · DECIDE AND PROMPT
# STEP 6 · Refuse when nothing relevant came back (the best score is too low): no LLM call.
# STEP 7 · "Which candidates know X?" has many answers: top-3 cannot hold them (Graph RAG / filters).
def should_refuse(query, hits):
    return False


def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{i}] ({h['source']}) {h['text']}" for i, h in enumerate(hits, 1))
    return f"CV passages:\n{ctx}\n\nQuestion: {question}"


# ================================================================= 6 · ANSWER (one full turn)
def rag(question, history, generate=True):
    """One RAG turn. Returns a trace of every step, for the Chat and the Verify tabs."""
    t = {"question": question}
    t0 = time.perf_counter()
    t["query"] = make_query(question, history)
    hits = retrieve(t["query"])
    t["search_ms"] = (time.perf_counter() - t0) * 1000
    t["refused"] = should_refuse(t["query"], hits)
    t["hits"] = [] if t["refused"] else hits
    t["prompt"] = "" if t["refused"] else build_prompt(question, hits)
    t1 = time.perf_counter()
    if t["refused"]:
        t["answer"] = REFUSAL
    elif generate:
        t["answer"] = LLM.chat(SYSTEM_PROMPT, t["prompt"], 300)
    else:
        t["answer"] = "(answers off: retrieval only)"
    t["answer_ms"] = (time.perf_counter() - t1) * 1000
    t["prompt_tokens"] = int(len((SYSTEM_PROMPT + t["prompt"]).split()) * 1.3)
    return t


# ================================================================= VERIFY (evaluation)
def load_questions(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def matches(source, expected):
    s = source.lower()
    return any(e.lower() in s or pathlib.Path(e).stem.lower() in s for e in expected)


def score(item, t):
    """Did retrieval find the expected CVs? Did it refuse when it should?"""
    expected = item.get("expected", [])
    if item.get("refuse"):
        ok = t["refused"] or REFUSAL.lower() in t["answer"].lower()
        return {"pass": ok, "hit": None, "recall": None, "rr": None, "note": "refused ✓" if ok else "answered, should refuse ✗"}
    found = {e for e in expected for h in t["hits"] if matches(h["source"], [e])}
    ranks = [h["rank"] for h in t["hits"] if matches(h["source"], expected)]
    recall = len(found) / len(expected) if expected else 0
    words = item.get("answer_contains", [])
    answer_ok = all(w.lower() in t["answer"].lower() for w in words) if words and t["answer"][:1] != "(" else None
    ok = recall == 1 and answer_ok is not False
    note = f"{len(found)}/{len(expected)} CVs found" + ("" if answer_ok is None else (" · answer ✓" if answer_ok else " · answer ✗"))
    return {"pass": ok, "hit": bool(ranks), "recall": recall, "rr": 1 / min(ranks) if ranks else 0, "note": note}


def run_verify(file, generate, progress=gr.Progress()):
    if file is None:
        return "Upload a questions JSON first.", None, gr.update(choices=[]), None, {}
    items, traces, rows = load_questions(file), {}, []
    for item in progress.tqdm(items, desc="Running questions"):
        t = rag(item["question"], item.get("history", []), generate)
        s = score(item, t)
        traces[item["id"]] = {"item": item, "trace": t, "score": s}
        rows.append(["✓" if s["pass"] else "✗", item["id"], item.get("type", ""), item["question"], s["note"],
                     ", ".join(h["source"] for h in t["hits"]) or "(refused)"])
    vals = [v["score"] for v in traces.values()]
    answerable = [s for s in vals if s["hit"] is not None]
    n = len(vals)
    summary = (f"### {sum(s['pass'] for s in vals)} / {n} passed\n\n"
               f"| Metric | Value |\n|---|---|\n"
               f"| Hit@{K} (an expected CV is in the top {K}) | {sum(s['hit'] for s in answerable) / max(len(answerable), 1):.0%} |\n"
               f"| Recall@{K} (share of expected CVs found) | {np.mean([s['recall'] for s in answerable]) if answerable else 0:.0%} |\n"
               f"| MRR (1 ÷ rank of the first right CV) | {np.mean([s['rr'] for s in answerable]) if answerable else 0:.2f} |\n"
               f"| Refusals correct | {sum(s['pass'] for s in vals if s['hit'] is None)} / {sum(1 for s in vals if s['hit'] is None)} |\n"
               f"| Avg. search / answer time | {np.mean([v['trace']['search_ms'] for v in traces.values()]):.0f} ms / "
               f"{np.mean([v['trace']['answer_ms'] for v in traces.values()]):.0f} ms |\n"
               f"| Avg. prompt size | {np.mean([v['trace']['prompt_tokens'] for v in traces.values()]):.0f} tokens |")
    out = pathlib.Path(__file__).parent / "verify_results.json"
    out.write_text(json.dumps({"settings": {"K": K, "chunk_size": CHUNK_SIZE}, "results": list(traces.values())},
                              indent=1, ensure_ascii=False, default=str))
    return summary, rows, gr.update(choices=list(traces), value=rows[0][1] if rows else None), str(out), traces


def show_detail(qid, traces):
    if not qid or qid not in traces:
        return "", None, "", ""
    v = traces[qid]
    t, item, s = v["trace"], v["item"], v["score"]
    head = (f"**{qid} · {item['question']}**\n\nExpected: `{', '.join(item.get('expected', [])) or 'refuse'}` · "
            f"Result: **{s['note']}** · Searched for: `{t['query']}`")
    table = [[h["rank"], h["source"], round(h["score"], 3), "✓ expected" if matches(h["source"], item.get("expected", [])) else "",
              h["text"][:300]] for h in t["hits"]]
    return head, table, f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{t['prompt']}", t["answer"]


# ================================================================= CHAT
def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    return " ".join(as_text(c) for c in content)


def respond(question, chat, generate):
    if not question.strip():
        return "", chat, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    past = [as_text(m["content"]) for m in chat if m["role"] == "user"]
    t = rag(question, past, generate)
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": t["answer"]}]
    query_md = (f"**You asked:** {question}\n\n**Searched for:** `{t['query']}`\n\n"
                f"Search {t['search_ms']:.0f} ms · answer {t['answer_ms']:.0f} ms · prompt {t['prompt_tokens']} tokens"
                + ("\n\n**Refused before calling the LLM.**" if t["refused"] else ""))
    table = [[h["rank"], h["source"], round(h["score"], 3), h["text"][:300]] for h in t["hits"]]
    return "", chat, query_md, table, f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{t['prompt']}", t["answer"]


# ================================================================= UI
print("Loading the models…")
EMBEDDER, LLM = get_embedder(), get_llm()
CHUNKS, VECTORS = [], None
STATUS = build_index()
print(STATUS)

with gr.Blocks(title="CV Chatbot") as demo:
    gr.Markdown(f"# CV Chatbot · naive RAG\n{STATUS} · K = {K} · chunk size {CHUNK_SIZE}")
    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(height=430)
                msg = gr.Textbox(placeholder="Ask about the candidates and press Enter", show_label=False)
                generate = gr.Checkbox(value=True, label="Generate the answer with the LLM (off = retrieval only, instant)")
                gr.Button("New chat").click(lambda: [], None, chatbot)
            with gr.Column(scale=6):
                query_out = gr.Markdown("Ask something to see how the answer is made.")
                with gr.Tab("Retrieved"):
                    hits_out = gr.Dataframe(headers=["#", "CV", "score", "chunk"], wrap=True, interactive=False)
                with gr.Tab("Prompt sent"):
                    prompt_out = gr.Code(language=None, label="Exactly what the LLM receives")
                with gr.Tab("Output"):
                    answer_out = gr.Markdown()
        msg.submit(respond, [msg, chatbot, generate], [msg, chatbot, query_out, hits_out, prompt_out, answer_out])
    with gr.Tab("✅ Verify"):
        traces = gr.State({})
        with gr.Row():
            qfile = gr.File(label="Questions JSON", file_types=[".json"], type="filepath")
            with gr.Column():
                vgen = gr.Checkbox(value=False, label="Also generate answers (slow on CPU; needed for answer_contains)")
                run = gr.Button("Run all questions", variant="primary")
                download = gr.File(label="Results (JSON)")
        summary = gr.Markdown()
        results = gr.Dataframe(headers=["pass", "id", "type", "question", "result", "retrieved CVs"], wrap=True, interactive=False,
                               column_widths=["6%", "6%", "11%", "37%", "16%", "24%"])
        gr.Markdown("### Open one question")
        pick = gr.Dropdown(label="Question", choices=[])
        d_head = gr.Markdown()
        d_hits = gr.Dataframe(headers=["#", "CV", "score", "expected?", "chunk"], wrap=True, interactive=False)
        with gr.Row():
            d_prompt = gr.Code(language=None, label="Prompt sent")
            d_answer = gr.Textbox(label="Output", lines=8)
        run.click(run_verify, [qfile, vgen], [summary, results, pick, download, traces])
        pick.change(show_detail, [pick, traces], [d_head, d_hits, d_prompt, d_answer])

if __name__ == "__main__":
    demo.launch(server_port=7862)
