"""
Lesson 02 · Chunking matters

PROBLEM  Fixed-size chunks cut sentences and sections in half, and a chunk that
         says only "Gateway timeout ... never retry" has lost the heading that
         told us which error code it is about.
IDEA     Split on natural boundaries (recursive / by headings) and keep the
         title and section with each chunk ("contextual chunks").
WATCH    How many chunks end mid-sentence, and whether ERR-5007 can be found
         when the code only appears in the heading.
TRY      Compare strategies on your own Markdown file in data/course/.

Run:  python lessons/02_chunking.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import BM25Index, VectorIndex, chunk_documents, get_embedder, load_company, load_course
from ragkit import show
from ragkit.evaluate import load_golden, run_eval

show.lesson("02", "Chunking matters",
            "Blind cuts break sentences; chunks lose the heading that gave them meaning.",
            "Split at paragraphs and headings, and attach title › section to every chunk.")

doc = next(d for d in load_course() if d.doc_id == "capstone-policy")

for strategy in ["fixed", "recursive", "headings"]:
    chunks = chunk_documents([doc], strategy=strategy, size=200)
    last_lines = [c.text.rstrip().splitlines()[-1] for c in chunks]
    broken = sum(1 for line in last_lines if not line.startswith("#") and line[-1:] not in ".!?:)|")
    show.step(f"Strategy '{strategy}': {len(chunks)} chunks, {broken} end mid-sentence")
    for c in chunks[:4]:
        show.info(f"   [{c.section or '-'}] {c.text[:95]!r}")

# --- Retrieval quality per strategy (course golden questions, top 3) ----------
show.step("Does it change retrieval? Hit rate on the course test questions (top 3, vector search)")
embedder = get_embedder()
golden = load_golden("course_questions.jsonl")
rows = []
for strategy in ["fixed", "recursive", "headings"]:
    chunks = chunk_documents(load_course(), strategy=strategy, size=300)
    index = VectorIndex(embedder, contextual=False).add(chunks)
    res = run_eval(lambda item: index.search(item["question"], 3), golden)
    rows.append({"strategy": strategy, "chunks": len(chunks), "hit@3": res["hit_rate"], "MRR": res["mrr"]})
show.table(rows, "Chunking strategy vs retrieval")
show.info("Fixed chunks can still 'hit' by luck, but the text handed to the model is broken.")

# --- Contextual chunks: the heading carries the key -------------------------
show.step("Contextual chunks: the error code lives only in the heading")
chunks = chunk_documents(load_company(), strategy="headings")
target = next(c for c in chunks if c.section == "ERR-5007")
show.info(f"   Chunk text alone : {target.text!r}")
show.info(f"   With context     : {target.with_context()!r}")
q = "Can we retry ERR-5007?"
for contextual in (False, True):
    bm25 = BM25Index(chunks, contextual=contextual)
    hits = bm25.search(q, 3)
    show.hits_table(hits, f"BM25 for '{q}' · contextual={contextual}",
                    expected=lambda c: c.section == "ERR-5007")

show.takeaway("Chunk at natural boundaries and keep 'title › section' with every chunk.\n"
              "From now on every lesson uses heading-based, contextual chunks.")
