# Glossary

**RAG (Retrieval-Augmented Generation)**: find relevant passages in your documents and give them to a language model with the question, so it answers from evidence.

**Chunk**: a piece of a document that is embedded and retrieved as one unit.

**Contextual chunk**: a chunk stored with its document title and section heading, so it still makes sense alone.

**Embedding**: a list of numbers representing the meaning of a text. Similar meanings get similar vectors.

**Cosine similarity**: how closely two vectors point in the same direction (1 = same meaning, 0 = unrelated).

**BM25**: classic keyword ranking. Scores exact word matches, rewarding rare words and penalizing long passages.

**Hybrid search**: running keyword (BM25) and vector search together and merging the results.

**RRF (Reciprocal Rank Fusion)**: merging ranked lists with score = Σ 1 / (60 + rank). Uses ranks only, so score scales don't matter.

**Bi-encoder**: embeds question and passage separately (fast, used for search).

**Cross-encoder / reranker**: reads question and passage together and outputs a relevance score (accurate, slow, used on a short list).

**Query rewriting**: turning a follow-up ("what about it?") into a standalone question.

**Multi-query / RAG-Fusion**: searching with several LLM-written phrasings and fusing the results.

**HyDE (Hypothetical Document Embeddings)**: searching with an LLM-drafted answer instead of the question.

**Metadata filter**: restricting retrieval by fields such as status, version, department or access role.

**Context management**: keeping the prompt small: summarizing old turns, dropping weak passages, compressing passages.

**Grounding**: the answer is supported by the retrieved sources. **Refusal**: saying "I don't know" when it isn't.

**Hit@k**: share of questions whose expected passage is in the top k. **MRR**: average of 1/rank of the first correct passage.

**Long context**: putting the whole (small) corpus in the prompt instead of retrieving.

**CAG (Cache-Augmented Generation)**: preloading a small, stable corpus into the model's context cache so questions skip retrieval.

**Agentic RAG**: an agent that decides when to search, which tools to call (search, live data, actions), checks the evidence and retries.
