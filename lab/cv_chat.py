"""
CHALLENGE · level 2 · Chat with the CVs, on YOUR pipeline

    python lab/cv_chat.py          (opens http://127.0.0.1:7861)

It imports lab/rag_lab.py, so every step you switched on (and every function you wrote for the
challenge) is what answers here. Left: the chat. Right: what was searched, which passages came
back and what the LLM read. Change ROLE to "hr" to see the private notes appear.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gradio as gr

import rag_lab as lab

lab.CORPUS = "cvs"
lab.setup()

EXAMPLES = ["Who are the strongest candidates for JOB-ML-01?", "Which candidates know Kubernetes?",
            "What did CAND-014 build at Cedar Payments?", "What salary does Lina Haddad expect?",
            "Which candidate has a pilot license?"]


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    return " ".join(as_text(c) for c in content)


def respond(question, chat, role, answers):
    lab.ROLE, lab.GENERATE_ANSWERS = role, answers
    past = [as_text(m["content"]) for m in chat if m["role"] == "user"]
    r = lab.ask(question, past)
    if r["refused"]:
        reply = lab.REFUSAL
    elif r["answer"]:
        reply = r["answer"]
    else:
        reply = "Passages found (answers are off, tick 'Generate answers'):\n\n" + \
                "\n".join(f"- {h.chunk.label}" for h in r["hits"])
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    rows = [[h.rank, h.chunk.meta.get("candidate_id", ""), h.chunk.label, round(h.score, 3),
             h.chunk.meta.get("access", "internal"), h.chunk.meta.get("status", "current")] for h in r["hits"]]
    info = f"**Searched for:** {r['query']}  \n**Passage tokens:** {r['tokens']} · **time:** {r['ms']:.0f} ms" + \
           ("  \n**Refused** before calling the LLM." if r["refused"] else "")
    return "", chat, info, rows, "\n\n---\n\n".join(r["texts"])


with gr.Blocks(title="CV Knowledge Desk") as demo:
    gr.Markdown("## CV Knowledge Desk · chat with the candidate CVs on your own pipeline")
    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=460)
            msg = gr.Textbox(placeholder="Ask about candidates, CVs or jobs…", show_label=False)
            gr.Examples(EXAMPLES, msg)
            with gr.Row():
                role = gr.Dropdown(["employee", "hr"], value="employee", label="Your role")
                answers = gr.Checkbox(value=False, label="Generate answers with the LLM (slow on CPU)")
                gr.Button("New chat").click(lambda: [], None, chatbot)
        with gr.Column(scale=6):
            info = gr.Markdown("Ask something to see how your pipeline answers.")
            table = gr.Dataframe(headers=["#", "candidate", "passage", "score", "access", "status"],
                                 interactive=False, wrap=True)
            prompt = gr.Code(language=None, label="What the LLM reads")
    msg.submit(respond, [msg, chatbot, role, answers], [msg, chatbot, info, table, prompt])

if __name__ == "__main__":
    demo.launch(server_port=7861)
