# Moving the index to a vector database

Lessons 01–13 use a tiny in-memory index (`ragkit/retrieval.py`) so you can read every line.
Lesson 14 adds persistence and incremental updates. In production you would use a vector
database. The concepts map one to one:

| Concept in ragkit | Chroma | Qdrant | pgvector (PostgreSQL) |
|-------------------|--------|--------|-----------------------|
| `VectorIndex.add` | `collection.upsert` | `client.upsert` | `INSERT ... ON CONFLICT` |
| `where=` filter | `where={"doc_id": ...}` | `query_filter=Filter(...)` | `WHERE meta->>'status' = 'current'` |
| cosine search | `hnsw:space = cosine` | `Distance.COSINE` | `ORDER BY embedding <=> $1` |
| chunk hash | store in metadata | store in payload | a `hash` column |

## Chroma (embedded, no server)

```python
import chromadb
client = chromadb.PersistentClient(path=".chroma")
col = client.get_or_create_collection("company", metadata={"hnsw:space": "cosine"})
col.upsert(ids=[c.id for c in chunks], embeddings=vectors.tolist(),
           documents=[c.text for c in chunks],
           metadatas=[{"doc_id": c.doc_id, "status": c.meta.get("status", "current"),
                       "access": c.meta.get("access", "internal")} for c in chunks])
res = col.query(query_embeddings=[q_vec.tolist()], n_results=3,
                where={"$and": [{"status": "current"}, {"access": {"$ne": "restricted"}}]})
```

## Qdrant (Docker: `docker run -p 6333:6333 qdrant/qdrant`)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
client = QdrantClient("localhost", port=6333)
client.recreate_collection("company", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.upsert("company", [PointStruct(id=i, vector=v.tolist(), payload={"text": c.text, **c.meta})
                          for i, (c, v) in enumerate(zip(chunks, vectors))])
hits = client.search("company", query_vector=q_vec.tolist(), limit=3,
                     query_filter=Filter(must=[FieldCondition(key="status", match=MatchValue(value="current"))]))
```

## pgvector (Docker: `docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg16`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
  id text PRIMARY KEY, doc_id text, section text, text text,
  meta jsonb, hash text, embedding vector(384)
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- search: current, non-restricted, top 3 by cosine distance
SELECT id, text, 1 - (embedding <=> $1) AS cosine
FROM chunks
WHERE meta->>'status' = 'current' AND meta->>'access' <> 'restricted'
ORDER BY embedding <=> $1
LIMIT 3;
```

Postgres also gives you keyword search in the same database (`tsvector`), so hybrid
search can be one SQL query plus RRF in Python.
