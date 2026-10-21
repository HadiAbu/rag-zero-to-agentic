# RAG Lab · 3 hours, one file, eleven steps

Students build a RAG system by **uncommenting one step at a time** in [`rag_lab.py`](rag_lab.py).
Every step replaces one of five small functions (`make_query`, `retrieve`, `postprocess`,
`build_context`, `should_refuse`), so the scoreboard at the bottom of the output shows exactly
what that one change did. The last two steps add Graph RAG and serve the result as an MCP tool
for Claude or Kiro. The session ends with a scored team challenge.

| File | What it is |
|------|-----------|
| `rag_lab.py` | The lab: STEP 1 on, STEPS 2–10 commented out, each with a *YOUR TURN* task |
| `mcp_server.py` | STEP 11: serves the student's own pipeline as an MCP server |
| `challenge.py` | Scores a pipeline on a question set (practice set included) |
| `challenge/practice.jsonl` | 6 practice questions, one of each type |
| `../app.py` | The RAG Lab chatbot (Gradio), for the live demo and for trying settings |

## How a step works

1. Run `python lab/rag_lab.py` and read the scoreboard.
2. Select the whole next STEP block and press **Ctrl+/** (**Cmd+/** on a Mac).
   Code lines start with `#` and switch on; explanation lines start with `##` and stay comments.
3. Run again and compare. Then do the step's **YOUR TURN** task.

Retrieval only runs by default, so each run takes seconds. Set `GENERATE_ANSWERS = True` to see
the LLM's answers too (about 25 s per question on a laptop CPU).

## Before the session (send to students 3 days before)

```bash
git clone https://github.com/mkabajah/rag-zero-to-agentic.git && cd rag-zero-to-agentic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio
python lab/rag_lab.py        # downloads the models once (about 3.2 GB) and prints a scoreboard
python app.py                # the chatbot opens at http://127.0.0.1:7860
```

Also install the assistant they will use in step 11: Claude Code, Claude Desktop or Kiro.

## The run sheet

Numbers are from a real run (MiniLM embeddings, ms-marco reranker; rewriting by rule).
Theirs will be close.

| Time | Step | Uncomment | What they should see | Ask the room |
|------|------|-----------|----------------------|--------------|
| 0:00 | Demo | – | You: `python app.py`, ask the 4 Build Day questions at checkpoint 1, then 7 | What changed between the two runs? |
| 0:15 | 1 · Naive RAG | – (on) | **2/4**: Q3 and Q4 miss · 743 passage tokens | Why can't "the above" be searched? |
| 0:22 | 2 · Chat history | STEP 2 | **4/4** · tokens up to 1,039 | It works. What did it cost? |
| 0:30 | 3 · Rewriting | STEP 3 | **4/4** · the rewritten queries are printed · +3–4 s per rewrite with the LLM | When would rewriting be wrong? |
| 0:45 | 4 · BM25 | STEP 4 | **4/4** · search takes ~0 ms · Q1 moves from #3 to #2 | What can BM25 never find? (YOUR TURN) |
| 0:55 | 5 · Hybrid (RRF) | STEP 5 | **4/4** | Students compute one RRF score by hand first |
| 1:05 | 6 · Rerank | STEP 6 | **4/4** · +150–230 ms per question | Is the time worth it? |
| 1:15 | 7 · Compress | STEP 7 | **4/4** · tokens 1,132 → **684** | Is every answer still cited? |
| 1:25 | 8 · Company docs | STEP 8, first only `CORPUS = "company"` | **2/6** · OUTDATED 2024 policy in the prompt · restricted text LEAKED | Who should fix this: the prompt or retrieval? |
| | | then the `FILTER` lines | **4/6** · version ✓ · protected ✓ | |
| 1:35 | Break | | | |
| 1:45 | 9 · Refusal gate | STEP 9 | **5/6** · CEO salary refused · but the spread-out question is refused too (0/4) | Can one threshold fix both? (0.317 vs 0.364: no) |
| 1:55 | 10 · Graph RAG | STEP 10 | **6/6** · graph path `ERR-5004, ERR-5007, ERR-4012, ERR-5100` → 4/4 found | Why can't top-3 ever find 4 answers? |
| 2:15 | 11 · MCP | – | Connect `mcp_server.py` (below), ask in Claude or Kiro | Why is the role an environment variable and not a tool argument? |
| 2:35 | Challenge | – | Teams tune, then you score them | |
| 2:55 | Results | – | Scores on screen, live round, takeaways | |

## Connect the MCP server (step 11)

The server imports `rag_lab.py`, so it serves whatever steps the student switched on.
The assistant's model writes the answer; the server only returns passages. The role is set in
the environment (`RAG_ROLE`), so the model cannot ask for more rights.

**Claude Code** (from the repo folder): the repo's `.mcp.json` already defines `rag-lab`.
Start `claude` in the repo and approve the server, or add it by hand:

```bash
claude mcp add --env RAG_ROLE=employee --transport stdio rag-lab -- .venv/bin/python lab/mcp_server.py
```

**Kiro**: the repo's `.kiro/settings/mcp.json` already defines `rag-lab`; open the repo as the workspace.

**Claude Desktop**: add the same block to `claude_desktop_config.json`, with absolute paths:

```json
{ "mcpServers": { "rag-lab": {
    "command": "/full/path/to/rag-zero-to-agentic/.venv/bin/python",
    "args": ["/full/path/to/rag-zero-to-agentic/lab/mcp_server.py"],
    "env": { "RAG_ROLE": "employee" } } } }
```

On Windows use `.venv\\Scripts\\python.exe`. If the server does not start, run
`python lab/mcp_server.py` in a terminal: it should wait silently (Ctrl+C to stop).

**Then ask in the chat:** "Can we retry ERR-5007?", "Which error codes must never be retried
automatically?", "What is the codename of the incident war room?". Change `RAG_ROLE` to
`security`, restart the server, and ask the last one again.

## The challenge · "Knowledge Desk" (20 min)

Teams of 3–4 tune `rag_lab.py` (steps, `K`, `GATE`, the graph rules, anything) to score as high
as possible on questions asked by an **employee**.

```bash
python lab/challenge.py                                   # practice set: teams try this
python lab/challenge.py lab/challenge/hidden.jsonl --team "Team 3"   # you, at the end
```

| Question type | Points |
|---------------|--------|
| exact, paraphrase, version | +2 right source in the prompt, +1 more if it is #1, −1 if an outdated version is also there |
| restricted | +3 nothing restricted reaches the prompt, −5 if it leaks |
| not in docs | +3 refused, −2 if passages were sent anyway |
| spread-out | +1 per required item found |

Ties go to fewer passage tokens. The full 10-step pipeline scores **19/19** on the practice set
but only **16/31** on the hidden set, so there is real room to win. Ideas that score: a better
refusal rule than one cosine threshold (try the reranker's score), graph rules that also catch
other phrasings, metadata filters, a different `K`.

**The hidden questions are not in this repo.** The instructor keeps `lab/challenge/hidden.jsonl`
(ignored by git) and copies it in only to score. Write your own in the same format as `practice.jsonl`.
