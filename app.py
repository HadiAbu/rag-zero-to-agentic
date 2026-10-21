"""
RAG Lab chatbot: a small Gradio app that shows how every answer is made.

Chat on the left. On the right, for the last answer: the query that was actually
searched, the ranked passages (with scores), the prompt sent to the model, and
the time each step took. Pick a checkpoint (1-7) to switch the pipeline, or set
each switch yourself.

Run:  pip install gradio
      python app.py            # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import time

import gradio as gr

from ragkit.context import compress, count_tokens
from ragkit.llm import ANSWER_SYSTEM, build_prompt, get_llm
from ragkit.pipelines import company_retriever, course_retriever
from ragkit.query import rewrite_query
from ragkit.retrieval import rrf

# ---------------------------------------------------------------- load once
print("Loading models and building the indexes (first run downloads the models)...")
COURSE = course_retriever()
COMPANY = company_retriever(embedder=COURSE.embedder, reranker=COURSE.reranker)
LLM = get_llm()
REFUSAL = "I don't know based on the provided documents."

# checkpoint -> pipeline switches (same order as the Build Day slides)
CHECKPOINTS = {
    "1 · Basic RAG": dict(history="none", retriever="vector", rerank=False, compress=False),
    "2 · + Chat context": dict(history="append", retriever="vector", rerank=False, compress=False),
    "3 · + Query rewriting": dict(history="rewrite", retriever="vector", rerank=False, compress=False),
    "4 · Keyword search (BM25)": dict(history="rewrite", retriever="bm25", rerank=False, compress=False),
    "5 · Hybrid search (RRF)": dict(history="rewrite", retriever="hybrid", rerank=False, compress=False),
    "6 · + Reranking": dict(history="rewrite", retriever="hybrid", rerank=True, compress=False),
    "7 · + Context management": dict(history="rewrite", retriever="hybrid", rerank=True, compress=True),
}
EXAMPLES = {
    "Course documents": ["Explain hybrid search and reranking.", "What are the advantages of the above?",
                         "Which one would find an exact policy ID?", "Can you compare them using the course documents?"],
    "Company documents": ["Can we retry ERR-5007?", "Which error codes must never be retried automatically?",
                          "How many days of annual leave do I get?", "What is the codename of the incident war room?"],
}


def can_see(role: str):
    """Access filter (lesson 10): restricted chunks only for their allowed roles."""
    def check(chunk) -> bool:
        if chunk.meta.get("access", "internal") != "restricted":
            return True
        allowed = chunk.meta.get("allowed_roles", [])
        return role in (allowed if isinstance(allowed, list) else [allowed])
    return check


def as_text(content) -> str:
    """Gradio 6 can store a message as a list of parts; join the text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    return " ".join(as_text(c) for c in content)


def current_only(chunk) -> bool:
    """Metadata filter (lesson 09): skip superseded document versions."""
    return chunk.meta.get("status", "current") != "superseded"


# ---------------------------------------------------------------- one turn of RAG
def run_pipeline(question, chat, corpus, role, history_mode, retriever, rerank, do_compress,
                 k, gate, current_filter, use_llm):
    R = COURSE if corpus == "Course documents" else COMPANY
    past = [as_text(m["content"]) for m in chat if m["role"] == "user"]
    times = {}

    # 1. what do we search for?
    t = time.perf_counter()
    if history_mode == "append":
        query, how = " ".join(past[-2:] + [question]), "question + last 2 user messages"
    elif history_mode == "rewrite":
        query, how = rewrite_query(past, question, LLM if use_llm else None)
    else:
        query, how = question, "question only"
    times["rewrite"] = (time.perf_counter() - t) * 1000

    # 2. filters (company documents carry access and version metadata)
    filters = []
    if corpus == "Company documents":
        filters.append(can_see(role))
        if current_filter:
            filters.append(current_only)
    where = (lambda c: all(f(c) for f in filters)) if filters else None

    # 3. retrieve
    t = time.perf_counter()
    pool = 10 if rerank else k
    if retriever == "vector":
        hits = R.vec.search(query, pool, where)
    elif retriever == "bm25":
        hits = R.bm25.search(query, pool, where)
    else:
        hits = rrf(R.vec.search(query, 10, where), R.bm25.search(query, 10, where), top=pool)
    times["search"] = (time.perf_counter() - t) * 1000
    candidates = hits

    t = time.perf_counter()
    if rerank:
        hits = R.reranker.rerank(query, hits, k)
    hits = hits[:k]
    times["rerank"] = (time.perf_counter() - t) * 1000 if rerank else 0

    # 4. relevance gate (lesson 12): refuse when nothing is close enough
    top_cos = R.vec.search(query, 1, where)
    top_cos = top_cos[0].score if top_cos else 0.0
    refused = gate > 0 and top_cos < gate

    # 5. build the prompt and answer
    texts = [h.chunk.with_context() for h in hits]
    if do_compress:
        texts = [tx.split("\n")[0] + "\n" + compress(query, h.chunk.text, 2) for tx, h in zip(texts, hits)]
    prompt = build_prompt(question if history_mode == "none" else query, texts)
    t = time.perf_counter()
    if refused:
        answer = REFUSAL + f"\n\n(Relevance gate: best similarity {top_cos:.2f} is below {gate:.2f}, so the model was not called.)"
    elif use_llm:
        answer = LLM.chat(ANSWER_SYSTEM, prompt, 250)
    else:
        answer = "Retrieval only (LLM switched off). The passages that would be sent are on the right."
    times["answer"] = (time.perf_counter() - t) * 1000 if not refused else 0

    if hits and not refused:
        answer += "\n\nSources: " + "; ".join(f"[{i}] {h.chunk.label}" for i, h in enumerate(hits, 1))

    # ---- inspector content
    query_md = (f"**You asked:** {question}\n\n**Searched for:** `{query}`\n\n*How:* {how}"
                + (f"\n\n*Filters:* role = `{role}`" + (", current versions only" if current_filter else "")
                   if corpus == "Company documents" else ""))
    used = {h.chunk.id for h in hits}
    table = [[h.rank, h.chunk.label, round(float(h.score), 4), "✓ sent to the LLM" if h.chunk.id in used else ""]
             for h in candidates[:10]]
    if rerank:
        table = [[h.rank, h.chunk.label, round(float(h.score), 3),
                  f"was #{h.details.get('before', '?')} before reranking"] for h in hits]
    total = sum(times.values())
    timing_md = (f"| Step | Time |\n|---|---|\n"
                 + "".join(f"| {name} | {ms:,.0f} ms |\n" for name, ms in times.items())
                 + f"| **total** | **{total:,.0f} ms** |\n\n"
                 f"Prompt size: **{count_tokens(ANSWER_SYSTEM + prompt)} tokens** "
                 f"({count_tokens(' '.join(texts))} of them passages) · best similarity {top_cos:.2f}")
    full_prompt = f"SYSTEM:\n{ANSWER_SYSTEM}\n\nUSER:\n{prompt}"
    return answer, query_md, table, full_prompt, timing_md


def respond(question, chat, *settings):
    if not question.strip():
        return "", chat, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    answer, query_md, table, prompt, timing = run_pipeline(question, chat, *settings)
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return "", chat, query_md, table, prompt, timing


def apply_checkpoint(name):
    c = CHECKPOINTS[name]
    return c["history"], c["retriever"], c["rerank"], c["compress"]


# ---------------------------------------------------------------- UI
with gr.Blocks(title="RAG Lab chatbot") as demo:
    gr.Markdown("# RAG Lab chatbot\nAsk about the course or company documents. "
                "The right side shows how the last answer was made. Change one setting at a time and ask again.")
    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=460, label="Chat")
            msg = gr.Textbox(placeholder="Ask a question and press Enter", show_label=False)
            with gr.Row():
                clear = gr.Button("New chat")
            corpus = gr.Radio(list(EXAMPLES), value="Course documents", label="Documents")
            examples = gr.Dataset(components=[msg], samples=[[q] for q in EXAMPLES["Course documents"]],
                                  label="Try these, in order")
        with gr.Column(scale=6):
            with gr.Accordion("Pipeline settings", open=True):
                checkpoint = gr.Dropdown(list(CHECKPOINTS), value="7 · + Context management",
                                         label="Checkpoint preset (sets the switches below)")
                with gr.Row():
                    history_mode = gr.Radio([("Question only", "none"), ("Add last messages", "append"),
                                             ("Rewrite with LLM", "rewrite")], value="rewrite", label="Follow-up questions")
                    retriever = gr.Radio([("Vector", "vector"), ("BM25", "bm25"), ("Hybrid (RRF)", "hybrid")],
                                         value="hybrid", label="Retriever")
                with gr.Row():
                    rerank = gr.Checkbox(value=True, label="Rerank (cross-encoder)")
                    do_compress = gr.Checkbox(value=True, label="Compress passages")
                    use_llm = gr.Checkbox(value=True, label="Generate answer with the LLM")
                with gr.Row():
                    k = gr.Slider(1, 8, value=3, step=1, label="Passages sent to the LLM (k)")
                    gate = gr.Slider(0, 0.6, value=0.0, step=0.05, label="Relevance gate (0 = off)")
                with gr.Row():
                    role = gr.Dropdown(["employee", "engineer", "security", "leadership"], value="employee",
                                       label="Your role (company documents)")
                    current_filter = gr.Checkbox(value=True, label="Current versions only")
            with gr.Tabs():
                with gr.Tab("1 · Query"):
                    query_out = gr.Markdown("Ask something to see what gets searched.")
                with gr.Tab("2 · Retrieved"):
                    table_out = gr.Dataframe(headers=["#", "passage", "score", "note"], wrap=True, interactive=False,
                                             column_widths=["7%", "53%", "13%", "27%"])
                with gr.Tab("3 · Prompt"):
                    prompt_out = gr.Code(language=None, label="Exactly what the LLM receives")
                with gr.Tab("4 · Time and size"):
                    timing_out = gr.Markdown()
            gr.Markdown(f"Models: `{COURSE.embedder.name}` · `{COURSE.reranker.name}` · `{LLM.name}`")

    settings = [corpus, role, history_mode, retriever, rerank, do_compress, k, gate, current_filter, use_llm]
    outputs = [msg, chatbot, query_out, table_out, prompt_out, timing_out]
    msg.submit(respond, [msg, chatbot] + settings, outputs)
    checkpoint.change(apply_checkpoint, checkpoint, [history_mode, retriever, rerank, do_compress])
    corpus.change(lambda c: (gr.Dataset(samples=[[q] for q in EXAMPLES[c]]), []), corpus, [examples, chatbot])
    examples.click(lambda s: s[0], examples, msg)
    clear.click(lambda: ([], "Ask something to see what gets searched.", None, "", ""), None,
                [chatbot, query_out, table_out, prompt_out, timing_out])

if __name__ == "__main__":
    demo.launch()
