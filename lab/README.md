# RAG Lab · 3 hours, one file, eleven steps

Students build a RAG system by **uncommenting one step at a time** in [`rag_lab.py`](rag_lab.py).
Every step replaces one of five small functions (`make_query`, `retrieve`, `postprocess`,
`build_context`, `should_refuse`), so the scoreboard at the bottom of the output shows exactly
what that one change did. The last two steps add Graph RAG and serve the result as an MCP tool
for Claude or Kiro. The session ends with a team challenge on candidate CVs: chat, MCP and a
**Knowledge Agent** in Claude Code or Kiro (brief: [`challenge/BRIEF.md`](challenge/BRIEF.md)).

**Give students [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md):** setup, team roles, the work loop and
copy-paste prompts for Claude Code and Kiro for every step and challenge level.

| File | What it is |
|------|-----------|
| `STUDENT_GUIDE.md` | The student handout: setup, team roles, the loop, prompt cards |
| `check_setup.py` | Run at home before the session: checks packages, models, data and the AI assistant |
| `../CLAUDE.md`, `../.kiro/steering/rag-lab.md` | Rules for the AI lab partner: explain first, small changes, run the scoreboard, no hardcoding |
| `rag_lab.py` | The lab: STEP 1 on, STEPS 2–10 commented out, each with a *YOUR TURN* task |
| `mcp_server.py` | STEP 11: serves the student's own pipeline as an MCP server |
| `challenge.py` | Scores a pipeline on a question set (CV practice set by default) |
| `challenge/BRIEF.md` | The challenge brief for the teams: five levels and the scoring |
| `challenge/cvs_practice.jsonl` | 7 practice questions on the CVs, one of each type |
| `cv_chat.py` | Challenge level 2: a Gradio chat over the CVs that runs the team's own pipeline |
| `../data/cvs/` | 20 invented candidates, 2 outdated CVs, private HR notes, 3 job posts |
| `../.claude/agents/knowledge-agent.md`, `../.kiro/agents/knowledge-agent.md` | The Knowledge Agent for Claude Code and Kiro |
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
python lab/check_setup.py    # downloads the models once (about 3.2 GB) and checks everything
python lab/rag_lab.py        # prints the first scoreboard
python app.py                # the chatbot opens at http://127.0.0.1:7860
```

Also install the assistant they will use in step 11: Claude Code, Claude Desktop or Kiro.

## The run sheet

Numbers are from a real run (MiniLM embeddings, ms-marco reranker; rewriting by rule).
Theirs will be close.

| Time | Step | Uncomment | What they should see | Ask the room |
|------|------|-----------|----------------------|--------------|
| 0:00 | RAG concepts | – | Slides: the RAG family, from naive to Graph and Agentic RAG | Which kind of RAG does your company need? |
| 0:30 | Demo | – | You: `python app.py`, ask the 4 Build Day questions at checkpoint 1, then 7 | What changed between the two runs? |
| 0:40 | 1 · Naive RAG | – (on) | **2/4**: Q3 and Q4 miss · 743 passage tokens | Why can't "the above" be searched? |
| 0:46 | 2 · Chat history | STEP 2 | **4/4** · tokens up to 1,039 | It works. What did it cost? |
| 0:52 | 3 · Rewriting | STEP 3 | **4/4** · the rewritten queries are printed · +3–4 s per rewrite with the LLM | When would rewriting be wrong? |
| 1:00 | 4 · BM25 | STEP 4 | **4/4** · search takes ~0 ms · Q1 moves from #3 to #2 | What can BM25 never find? |
| 1:07 | 5 · Hybrid (RRF) | STEP 5 | **4/4** | Students compute one RRF score by hand first |
| 1:15 | 6 · Rerank | STEP 6 | **4/4** · +150–230 ms per question | Is the time worth it? |
| 1:22 | 7 · Compress | STEP 7 | **4/4** · tokens 1,132 → **684** | Is every answer still cited? |
| 1:30 | Break | | | |
| 1:40 | 8 · Company docs | STEP 8, first only `CORPUS = "company"` | **2/6** · OUTDATED 2024 policy in the prompt · restricted text LEAKED | Who should fix this: the prompt or retrieval? |
| | | then the `FILTER` lines | **4/6** · version ✓ · protected ✓ | |
| 1:47 | 9 · Refusal gate | STEP 9 | **5/6** · CEO salary refused · but the spread-out question is refused too (0/4) | Can one threshold fix both? (0.317 vs 0.364: no) |
| 1:52 | 10 · Graph RAG | STEP 10 | **6/6** · graph path `ERR-5004, ERR-5007, ERR-4012, ERR-5100` → 4/4 found | Why can't top-3 ever find 4 answers? |
| 1:58 | 11 · MCP | – | Connect `mcp_server.py` (below), ask in Claude or Kiro | Why is the role an environment variable and not a tool argument? |
| 2:05 | Challenge | CHALLENGE block | Teams follow `challenge/BRIEF.md`: CVs → chat → smart → MCP → Knowledge Agent | |
| 2:50 | Results | – | Hidden questions scored, one live brief per top team, takeaways | |

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

## The challenge · "The Knowledge Agent" (45 min)

Teams follow [`challenge/BRIEF.md`](challenge/BRIEF.md): point the pipeline at the candidate CVs
(`CORPUS = "cvs"`), chat with them (`python lab/cv_chat.py`), make retrieval smart enough for
hiring questions, connect the `cv-desk` MCP server and drive it with the **Knowledge Agent** in
Claude Code or Kiro.

```bash
python lab/challenge.py                                         # teams: 7 practice questions on the CVs
python lab/challenge.py lab/challenge/hidden.jsonl --team "Team 3"   # you, at the end
```

| Question type | Points |
|---------------|--------|
| exact, paraphrase, version | +2 right CV in the prompt, +1 more if it is #1, −1 if the outdated CV is also there |
| restricted | +3 nothing private reaches the prompt, −5 if it leaks |
| not in the CVs | +3 refused, −2 if passages were sent anyway |
| spread-out | +1 per required item found |
| match | +1 per right candidate in the prompt, −1 per wrong one |

Then each team's agent answers one hiring brief live (15 points: right people 6, citations 3,
honest gaps 3, safe and fair 3). The 10-step pipeline scores **11/21** on the practice set and
**29/41** on the hidden set; a reference solution (named-candidate search, metadata and skill
filters, a reranker-score refusal) reaches 20/21 and 37/41.

**The hidden questions and the reference solution are not in this repo.** The instructor keeps
`lab/challenge/hidden.jsonl` (ignored by git) and copies it in only to score. The older
company-documents challenge still works: `python lab/challenge.py --corpus company`.
