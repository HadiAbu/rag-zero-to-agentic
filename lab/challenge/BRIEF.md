# The challenge · build the Knowledge Agent (45 min, teams of 3–4)

A hiring team has **20 candidate CVs** and **3 open jobs** (`data/cvs/`). They want to ask
questions in plain language and get answers they can trust: "Who knows Kubernetes?",
"What did CAND-014 build?", "Shortlist three people for JOB-ML-01".

You already built the RAG pipeline in `lab/rag_lab.py`. Now point it at the CVs, make it smart,
serve it over MCP and put a **Knowledge Agent** on top of it in Claude Code or Kiro.

All the people and companies in the CVs are invented. Some files are traps:

| Trap | Where |
|------|-------|
| Outdated CVs (`status: superseded`) | Rania Khoury (2024) and Tamar Klein (2023) have an older CV that says something different |
| Private HR notes (`access: restricted`, `allowed_roles: hr`) | `hr-private/`: salary, phone, notice period, recruiter notes |
| Answers spread over many CVs | "Which candidates know Kubernetes?" has 4 answers; top-3 cannot hold them |
| Filters, not similarity | "in Haifa with at least 5 years": the answer is in the metadata (`location`, `years`) |
| Things nobody has | "Who has a pilot license?" must be refused |

## The five levels

| Level | Time | Do this | Check |
|-------|------|---------|-------|
| 1 · Load | 5 min | In `rag_lab.py`, uncomment `CORPUS = "cvs"` in the CHALLENGE block and run it | The CV scoreboard (6 questions) |
| 2 · Chat | 5 min | `python lab/cv_chat.py`: chat with the CVs on your pipeline. Switch the role to `hr` | Private notes appear only for `hr` (after STEP 8's filter) |
| 3 · Make it smart | 15 min | Replace functions in the CHALLENGE block: a CV graph, filters from the question, a better refusal | `python lab/challenge.py` (7 practice questions) |
| 4 · MCP | 5 min | Connect the `cv-desk` server (already in `.mcp.json` and `.kiro/settings/mcp.json`). Bonus: add your own tool in `lab/mcp_server.py` | The assistant lists `find_candidates` and `list_jobs` |
| 5 · Knowledge Agent | 10 min | Use the agent in `.claude/agents/knowledge-agent.md` (Claude Code) or `.kiro/agents/knowledge-agent.md` (Kiro). Improve its instructions | It shortlists with cited evidence and refuses private data |

**Claude Code:** start `claude` in the repo, approve `cv-desk`, then ask
"Use the knowledge-agent to shortlist three candidates for JOB-ML-01" (`/agents` lists it).
**Kiro:** open the repo, choose `knowledge-agent` in the agent picker of the chat, ask the same.

## Scoring

**A · Hidden questions (automatic, about 40 points).** At the end the instructor runs
`python lab/challenge.py hidden.jsonl --team "<name>"` on your `rag_lab.py`, always as an interviewer.

| Question type | Points |
|---------------|--------|
| exact, paraphrase, version | +2 the right CV in the prompt, +1 if it is #1, −1 if the outdated CV is also there |
| restricted | +3 nothing private reaches the prompt, −5 if it leaks |
| not in the CVs | +3 refused, −2 if passages were sent anyway |
| spread-out | +1 per required item found |
| match (a list of candidates) | +1 per right candidate in the prompt, **−1 per wrong one** |

**B · The live brief (judged, 15 points).** Your agent gets one hiring brief on the projector.

| What we look at | Points |
|-----------------|--------|
| The right people in the shortlist | 6 |
| Every claim cited with a candidate ID and source | 3 |
| Honest gaps: says what is missing instead of inventing | 3 |
| Safe and fair: no private HR data, no age, gender, religion or origin used | 3 (−5 if broken) |

**C · Bonus +3:** a new MCP tool that your agent actually calls in the brief.

Ties go to fewer passage tokens. For reference: the 10-step pipeline from the lab scores
**11 / 21** on the practice set, so there is a lot of room.

## Ideas that score

- `FILTER` from STEP 8 already hides superseded CVs and HR notes. Is it switched on?
- A **named candidate** ("CAND-014", "Lina Haddad") should search only that person's CV.
- "Which / who … candidates" questions want **every** match, not top-3: filter on `chunk.meta`
  (`location`, `years`) and on the Skills section, like `find_candidates` does in the MCP server.
- One cosine threshold refuses good questions. The reranker's score separates "not in the CVs"
  from real questions much better: print it and pick a number.
- The agent is only as good as its tools: a `match_job(job_id)` tool that checks every must-have
  saves it ten tool calls.
