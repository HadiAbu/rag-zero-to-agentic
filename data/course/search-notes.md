---
title: Search Techniques (Course Notes)
doc_id: search-notes
course: AI Engineering
version: 3
updated: 2026-09-01
---
# Search Techniques (Course Notes)

## 1. Vector search

Vector search turns every passage and every question into an embedding, a list of numbers that captures meaning. Search becomes geometry: we return the passages whose embeddings are closest to the question's embedding, usually by cosine similarity. Because it compares meaning, vector search matches paraphrases even when the words differ, for example "holiday" and "vacation". Its weakness is rare exact tokens: product codes, error codes and IDs such as POL-204 are often blurred into "something that looks like a code".

## 2. Hybrid search

Hybrid search runs BM25 keyword search and vector search in parallel, then merges the two ranked lists. The usual merge is reciprocal rank fusion (RRF): every passage gets 1 / (60 + rank) from each list it appears in, and the totals are sorted. RRF only uses ranks, so it does not matter that BM25 scores and cosine scores live on different scales.

## 3. Reranking

A reranker is a cross-encoder: it reads the question and one candidate passage together and outputs a relevance score. Because it sees both texts at once it is more accurate than comparing two separate embeddings, but it is also slower. That is why reranking runs only on a short list, typically the top 20 to 50 results from first-stage retrieval, and then keeps the best 3 to 5.

## 4. Advantages of hybrid search and reranking

The main advantage of hybrid search is coverage: keyword search catches exact terms such as policy IDs, error codes and names, while vector search catches paraphrases and synonyms. The main advantage of reranking is precision: it puts the best evidence first, so fewer passages need to go into the prompt, which lowers cost and reduces distraction for the model.

## 5. Limitations

Running two indexes means more storage and two things to keep in sync whenever documents change. A second-stage model adds latency to every question, roughly one model pass per candidate passage. No search technique can find information that is not in the documents.
