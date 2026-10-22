# RAG Build Day · student guide

Today you build a RAG system yourself, step by step, with an AI partner (Claude Code or Kiro).
The AI writes code fast. **The scoreboard decides if the code is better.** You decide what to try.

## Before the session (at home, 20 minutes)

```bash
git clone https://github.com/mkabajah/rag-zero-to-agentic.git && cd rag-zero-to-agentic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio
python lab/check_setup.py     # downloads the models once (about 3.2 GB) and checks everything
```

Install **Claude Code** or **Kiro** and open this folder in it once. The repo already tells the AI
how to behave in this lab (`CLAUDE.md`, `.kiro/steering/rag-lab.md`): explain first, small changes,
run the scoreboard after every change, no hardcoded answers.

## Your team (3 people, swap roles every 2 steps)

| Role | Does |
|------|------|
| **Driver** | Has the keyboard: uncomments, runs, edits |
| **Navigator** | Talks to Claude or Kiro: writes the prompts, reads the answer out loud |
| **Checker** | Says the **prediction** before each run, compares the scoreboard after, keeps the team's log |

## The loop for every step (about 6 minutes)

1. **Predict.** Before running, the Checker writes down: which questions pass, and the passage tokens.
2. **Uncomment** the STEP block (select it, `Ctrl+/` or `Cmd+/`) and run `python lab/rag_lab.py`.
3. **Compare** with your prediction. Were you right? Why not?
4. **Explain back.** Ask your AI partner the step's *Ask* prompt below. Can every one of you explain the change in one sentence?
5. **YOUR TURN.** Do the small task at the end of the step, with the *Try* prompt if you need it.

Stuck for more than 3 minutes? Raise the red card, or type 🟥 and your team name in the chat.

## Prompt cards for the steps

Copy a prompt into Claude Code or Kiro. Change it; it is a starting point, not a recipe.

| Step | Ask (understand) | Try (YOUR TURN) |
|------|------------------|-----------------|
| 1 · Naive RAG | "Explain in 4 lines what make_query, retrieve and postprocess do in lab/rag_lab.py. Which of the 4 questions will fail and why? Don't change code." | "Set K to 1, then 5, run the lab each time and make a small table: K, correct, passage tokens." |
| 2 · Chat history | "Why does adding the last 2 user messages fix Q3 and Q4? What does it cost?" | "Change -2 to -1 and -4. Run each and tell me which question breaks." |
| 3 · Rewriting | "Show me REWRITE_SYSTEM in ragkit/query.py and explain what the model is asked to do." | "Suggest one sentence to add to the rewrite instruction. Explain the risk, then let us decide." |
| 4 · BM25 | "Explain BM25 with the ERR-5007 example: why does a rare word count more?" | "Add a paraphrase question to QUESTIONS['course'] that BM25 should fail on. Run and show the scoreboard." |
| 5 · Hybrid (RRF) | "Compute the RRF score of one chunk by hand with me, using its two ranks." | "Change 60 to 1 and to 1000, run both, explain the change in order." |
| 6 · Rerank | "What is the difference between a bi-encoder and a cross-encoder, in 3 sentences?" | "Print how many milliseconds postprocess adds per question. Is it worth it?" |
| 7 · Compress | "What does compress() keep and what does it drop? Show one example." | "Set GENERATE_ANSWERS = True for one question. Is the answer still cited?" |
| 8 · Company docs | "Why must the version and role filter be in retrieval and not in the prompt?" | "Run with ROLE = 'security'. Which result changes, and is that correct?" |
| 9 · Refusal | "Print best[0].score for the 6 company questions. Is there one GATE that works for all?" | "Try 3 GATE values and make a table of what passes." |
| 10 · Graph RAG | "Explain extract, build_graph and graph_search in plain words." | "Make graph_search also answer 'Which codes can't be retried?'. Run and check nothing else broke." |
| 11 · MCP | "Which tools does the rag-lab server offer? Call search_docs for 'Can we retry ERR-5007?'" | Set RAG_ROLE to security, restart the server, ask about the war room again. |

## The challenge · the Knowledge Agent (45 minutes)

Full brief: [`challenge/BRIEF.md`](challenge/BRIEF.md). You work on the **candidate CVs** in `data/cvs`.

**Rules of the game**
- Your score comes from **hidden questions**. Hardcoding practice answers scores nothing there.
- Every idea is one small function in the CHALLENGE block. Run `python lab/challenge.py` before and after.
- Keep a log: idea → practice score. The best log gets a shout-out even if it does not win.

**Prompt cards for the levels**

**Level 1 · Load:** "Uncomment CORPUS = 'cvs' in the CHALLENGE block, run the lab, and explain each ✗ on the scoreboard."

**Level 2 · Chat:** run `python lab/cv_chat.py` and ask the examples. Switch the role to `hr`. What appears, and why?

**Level 3 · Make it smart** (pick one idea at a time):
- "In the CHALLENGE block, write a retrieve() that searches only one candidate's chunks when the question names a candidate (a CAND-xxx ID or a name from chunk.meta['name']), and otherwise uses the previous retrieve. Run python lab/challenge.py before and after."
- "For 'which / who … candidates' questions, return ALL matching candidates instead of top-3: read the city and 'at least N years' from the question and filter on chunk.meta. Don't hardcode cities from the practice file: take them from the documents."
- "Print the reranker score of the top passage for the 7 practice questions. Then write a should_refuse() that uses it instead of cosine. Explain the threshold you pick."
- "Build a skills graph: from each CV's Skills section, add (candidate, has_skill, skill) to the graph and answer 'Which candidates know X?' with every match."

**Level 4 · MCP:** start Claude Code or Kiro in the repo and approve the `cv-desk` server. Then:
"Add an MCP tool match_job(job_id) to lab/mcp_server.py that returns every candidate with the must-haves they meet and the evidence for each. Restart cv-desk and call it for JOB-BE-02."

**Level 5 · Knowledge Agent:**
- Claude Code: "Use the knowledge-agent to shortlist three candidates for JOB-ML-01."
- Kiro: pick **knowledge-agent** in the agent picker, ask the same.
- Then improve it: "Edit the knowledge-agent instructions so every shortlist also gives one interview question per candidate, and it says which must-have is weakest."

**Test your agent like a reviewer:** Does every claim have a CAND-ID and a source? Did it refuse
salary questions? Did it use an outdated CV? Did it invent anything?

## At the end

1. The instructor runs the hidden questions on your `lab/rag_lab.py`.
2. Each team's agent answers one live hiring brief on the projector, and the room checks the citations.
3. **Red team round:** write one tricky question for the team next to you. Can their agent survive it?
