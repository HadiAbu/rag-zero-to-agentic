# Teacher guide

## Before the session

- Run `python run_all.py` once on the classroom machine so all models are downloaded (about 3.2 GB in total, mostly the local LLM).
- If the Wi-Fi is weak, install Ollama and `ollama pull llama3.2:3b` beforehand, or ask students to set `HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct` (1 GB).
- Make the terminal wide (120+ columns) so the tables are readable: `COLUMNS=120 python lessons/05_keyword_search_bm25.py`.

## Suggested 3-hour plan

| Time | Block | Lessons | What to point at |
|------|-------|---------|------------------|
| 0:00–0:10 | RAG map | 01 | The four steps and the retrieved chunks on screen |
| 0:10–0:35 | Build basic RAG | 01, 02 | Broken chunks; ERR-5007 unfindable without the heading |
| 0:35–0:55 | Retrieval failure clinic | 03, 05, 12 | Follow-ups, exact codes, paraphrases, unanswerable questions |
| 0:55–1:25 | Hybrid search and reranking | 04, 06, 07 | Rewritten query beside the original; RRF table; before/after ranks |
| 1:25–1:35 | Break | | |
| 1:35–2:05 | RAG, long context and CAG | 17, 11 | Tokens and ms per approach; the cost table (replace the example rates) |
| 2:05–2:25 | When to use tools | 16, 10 | Live data comes from tools; permissions are filters |
| 2:25–2:50 | Evaluation and cost | 13, 09 | The scoreboard; stale versions |
| 2:50–3:00 | Design review | 15 | Each group defends one design with two scoreboard rows |

## The Build Day question sequence

Asked in one conversation, in order (data/golden/conversation.json):

1. "Explain hybrid search and reranking." (sets the topic)
2. "What are the advantages of the above?" (exposes question-only retrieval)
3. "Which one would find an exact policy ID?" (keyword search)
4. "Can you compare them using the course documents?" (several passages, reranking, context size)

Lesson 13 part B prints the scoreboard for these four questions at all seven checkpoints.

## Teaching points that land

- A fluent answer can still be wrong (lesson 12 compares a loose prompt with a grounded one; small models readily add details that are not in the sources). Always look at the retrieved passages to tell a search failure from a generation failure.
- Similarity search finds candidates; reranking reorders them. Their scores are on different scales (cosine 0.6 vs cross-encoder −10 to +10), so compare ranks and evidence, not numbers.
- Security is a retrieval filter, never a prompt instruction.
- RAG supplies information; tools fetch live state or take actions.
- RAG is not automatically cheaper: use the provider's current price sheet in lesson 17.

## Known behaviour of the small local model

The default free LLM (Qwen2.5-1.5B on CPU) is good enough to show every idea, but:

- It sometimes ignores the instruction to cite `[1]`.
- Query rewriting falls back to a simple rule when the model's rewrite drops the topic (the lesson prints which method was used).
- The agent lesson uses rules for routing unless `RAG_ROUTER=llm`; a 3B+ model via Ollama routes well on its own.

These are good discussion points, not bugs: they are exactly why production systems add guardrails.
