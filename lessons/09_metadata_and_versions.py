"""
Lesson 09 · Metadata: stale versions and filters          (company RAG)

PROBLEM  Companies keep old documents. The 2024 leave policy (22 days) and the
         2026 policy (25 days) look almost identical to a retriever, and the
         old one can rank first. The answer is then confidently wrong.
IDEA     Store metadata with every chunk (status, version, effective_date,
         department) and FILTER before ranking: only current documents.
WATCH    The same question with and without the filter, and the answers.
TRY      Filter by department: where=lambda c: c.meta["department"] == "Finance".

Run:  python lessons/09_metadata_and_versions.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import answer, get_llm, show
from ragkit.pipelines import company_retriever

show.lesson("09", "Metadata and versions",
            "An outdated policy competes with the current one and can win.",
            "Filter on metadata (status=current) before ranking, and show the version in citations.")

R = company_retriever()
llm = get_llm()
show.setup_info(R.embedder, llm)

show.step("Metadata stored with each chunk")
for c in R.chunks:
    if c.doc_id.startswith("leave-policy") and c.section.startswith("1."):
        show.info(f"   {c.label:40s} status={c.meta.get('status')}  version={c.meta.get('version')}  "
                  f"effective={c.meta.get('effective_date')}")

current_only = lambda c: c.meta.get("status", "current") == "current"
for q in ["How many days of annual leave do I get?", "How many unused vacation days can I carry over?"]:
    show.step(q)
    plain = R.hybrid(q, 3)
    show.hits_table(plain, "No filter", expected=lambda c: c.meta.get("status") == "superseded", mark="⚠️ OUTDATED")
    show.answer_box(answer(q, plain, llm), "Answer without filter")
    filtered = R.hybrid(q, 3, where=current_only)
    show.hits_table(filtered, "Filter: status == current")
    show.answer_box(answer(q, filtered, llm), "Answer with filter")

show.takeaway("Ingest metadata, not just text. Filter by status, date, department or product\n"
              "before ranking, so outdated or irrelevant documents never compete.")
