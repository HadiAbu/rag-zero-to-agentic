# The CV Chatbot · build it, then prove it

A Gradio chatbot over candidate CVs. You start from **naive RAG** (`app.py`, about 250 lines)
and improve it one step at a time with Claude Code or Kiro. After every step you run the
**Verify** tab: same questions, same CVs, and the numbers decide whether your change helped.

```bash
git pull                                   # get the latest version
pip install pypdf python-docx gradio       # PDF and Word reading, the UI
# put the CVs your instructor sent you in challenge/cvs/   (ignored by git: never pushed)
python challenge/app.py                      # → http://127.0.0.1:7862
```

## What you see

| Tab | Shows |
|-----|-------|
| 💬 Chat | The answer, and on the right: what was **searched**, the **retrieved** chunks with their scores, the exact **prompt sent**, and the **output** with timings |
| ✅ Verify | Upload a questions JSON, run them all: Hit@K, Recall@K, MRR, refusals, time, prompt size, and one row per question you can open to see its retrieval, prompt and output. The results are saved to `challenge/verify_results.json` |

Answers take 20–60 s each on a laptop CPU. Untick **Generate the answer** to test retrieval instantly.

## The questions JSON

```json
{"questions": [
  {"id": "q01", "type": "exact", "question": "What did Fadi Khalil build at Cedar Payments?",
   "expected": ["Fadi_Khalil"], "answer_contains": ["onboarding"]},
  {"id": "q05", "type": "follow-up", "question": "And where did he study?",
   "history": ["Who made a website much faster to load?"], "expected": ["Eitan_Barak"]},
  {"id": "q08", "type": "not in CVs", "question": "Which candidate has a pilot license?", "refuse": true}
]}
```

- `expected`: the CV file names (or a part of them) that must be retrieved. Several = all of them.
- `refuse: true`: the CVs do not contain the answer; the bot must say it does not know.
- `history`: earlier questions in the same chat (for follow-ups).
- `answer_contains`: words the answer must contain (only checked when answers are on).

`questions_example.json` works on the practice CVs in `data/cvs`. Write 3 questions of your own:
it is the best way to learn what your bot cannot do yet.

## The steps (the menu)

Do them in any order, **one at a time**, and run Verify after each. Every step is marked
`# STEP n` in `app.py`. The prompts are a start: change them.

| Step | Fixes | Prompt for Claude Code or Kiro | Watch in Verify |
|------|-------|--------------------------------|-----------------|
| 1 · Follow-ups | "And where did he study?" searches for nothing | "In challenge/app.py, change make_query so a follow-up question is rewritten into a full question using the chat history. You can use ragkit.query.rewrite_query. Then run the Verify tab logic on questions_example.json and show me the follow-up row before and after." | The `follow-up` rows |
| 2 · Better chunks | A chunk from page 2 does not say whose CV it is | "Change chunk() to split each CV by its sections (Summary, Experience, Skills, Education…) and put the candidate's name in front of every chunk. Show me 3 chunks before and after." | Hit@K, MRR |
| 3 · Hybrid search | Exact names, tools and certificates are missed | "Add BM25 keyword search next to the vector search and merge the two lists with Reciprocal Rank Fusion (score = sum of 1/(60+rank)). ragkit has a BM25Index you can read for the formula." | `exact` rows |
| 4 · Rerank | The right CV is found, but not first | "Get 20 candidates, rerank them with the cross-encoder in ragkit.Reranker (its .model.predict takes (question, text) pairs) and keep the top K. Print the time it adds." | MRR, time |
| 5 · Metadata and filters | "In Haifa", "5+ years" are guessed by similarity | "At load time, extract each CV's name, city and years of experience into metadata. In retrieve, read a city or a minimum of years from the question and filter before searching." | `list` rows |
| 6 · Refusal | It always answers, even when no CV fits | "Make should_refuse return True when the best retrieved chunk is not relevant enough. Print the top score for every Verify question first, then pick a threshold and explain it." | Refusals correct |
| 7 · Lists (Graph RAG) | "Which candidates know X?" has more answers than K | "For 'which / who … candidates' questions, return ALL matching CVs, not top-K: build a small graph (candidate → skill, candidate → city, candidate → company) at load time and answer from it." | Recall@K on `list` rows |
| 8 · MCP + agent | Use it from Claude or Kiro | "Write challenge/mcp_server.py with FastMCP exposing search_cvs(question) and get_cv(name), using the functions in challenge/app.py. Then add it to .mcp.json as cv-chatbot." | Ask your agent for a shortlist |

## Rules

- **One change, then Verify.** Keep a log: step → passed, Hit@K, MRR, time.
- **No hardcoding** names or answers from the questions file. Your instructor tests with questions you have not seen.
- **The CVs are real people's data.** Keep them on your laptop, out of git and out of any public place,
  and judge candidates only on job-relevant facts.
