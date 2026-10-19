# Data

| Folder | What | Used in |
|--------|------|---------|
| `course/` | Course documents: search techniques notes, capstone policy, team brief, POL-204 team policy, FAQ, model options | 01–08, 11, 13, 14, 16, 17 |
| `company/` | Company documents with metadata (department, access, allowed_roles, version, status, effective_date) | 05–07, 09, 10, 12, 13, 15, 16 |
| `golden/` | Test questions with the expected source document and a snippet the right chunk must contain | 02, 03, 04, 06, 07, 13 |
| `live/` | Mock live system (student records) for the agent; created tickets are written here | 16 |

## The slide deck's D1–D8

The RAG Build Day slides refer to course documents D1–D8. They are these files:

| Slide ID | File › section |
|----------|----------------|
| D1 | `course/search-notes.md` › 2. Hybrid search |
| D2 | `course/search-notes.md` › 3. Reranking |
| D3 | `course/search-notes.md` › 4. Advantages of hybrid search and reranking |
| D4 | `course/search-notes.md` › 1. Vector search |
| D5 | `course/capstone-policy.md` › 2. Deadline and late work |
| D6 | `course/team-brief.md` › 1. Identifiers (A17, POL-204) |
| D7 | `course/team-policy.md` › 1. Shared responsibility |
| D8 | `course/course-faq.md` › 5. Office hours |

The slides use illustrative scores; the lessons compute real ones on these files.

## Front matter

Every document starts with a small header. Add your own files in the same format:

```
---
title: Remote Work Policy
doc_id: remote-work
department: HR
access: internal            # internal | restricted
allowed_roles: security, leadership   # only for restricted documents
version: 3
status: current             # current | superseded
effective_date: 2025-06-01
---
```

## Golden questions (`golden/*.jsonl`)

One JSON object per line: `question`, `expected_docs` (list of `doc_id`s; empty = should be refused),
`contains` (text the right chunk must contain), `type` (direct, paraphrase, exact, multi, stale, acl,
table, unanswerable) and, for company questions, the asking user's `role`.

All documents are fictional and written for teaching.
