"""
Lesson 16 · Agentic RAG: retrieval becomes one tool among many

PROBLEM  "Am I allowed to take the quiz? I'm student s-1002." The rule is in a
         document (80% attendance), but the student's attendance is LIVE data.
         RAG must never guess live data, and some requests are actions
         ("open a ticket"), not questions.
IDEA     An agent loop:
           1. a router picks tools (document search, student record, ticket, none)
           2. it runs them and checks the evidence (retries with a new query if weak)
           3. it answers from the tool results, citing them
WATCH    The trace: which tools were chosen, what came back, retries.
         With a small model the router may fall back to simple rules; try
         RAG_LLM=ollama (llama3.2:3b or qwen2.5:7b) to see an LLM route on its own.

Run:  python lessons/16_agentic_rag.py
"""
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit import get_llm, show
from ragkit.documents import DATA
from ragkit.pipelines import company_retriever, course_retriever
from ragkit.query import multi_query

show.lesson("16", "Agentic RAG",
            "Some questions need live data or actions, which documents can't provide.",
            "Let a router choose tools, check the evidence, retry if weak, then answer.")

COURSE = course_retriever()
COMPANY = company_retriever(embedder=COURSE.embedder, reranker=COURSE.reranker)
llm = get_llm()
show.setup_info(COURSE.embedder, llm, COURSE.reranker)
STUDENTS = json.loads((DATA / "live" / "students.json").read_text())
TICKETS = DATA / "live" / "tickets_created.json"


# ------------------------------------------------------------------ tools
def search_course(query: str) -> dict:
    hits = COURSE.hybrid_rerank(query, 2)
    return {"evidence": [h.chunk.text for h in hits], "sources": [h.chunk.label for h in hits],
            "strength": COURSE.vector(query, 1)[0].score}


def search_company(query: str) -> dict:
    current = lambda c: c.meta.get("status") == "current" and c.meta.get("access") != "restricted"
    hits = COMPANY.hybrid_rerank(query, 2, where=current)
    return {"evidence": [h.chunk.text for h in hits], "sources": [h.chunk.label for h in hits],
            "strength": COMPANY.vector(query, 1, where=current)[0].score}


def student_record(student_id: str) -> dict:
    rec = STUDENTS.get(student_id)
    text = (f"Student record {student_id} ({rec['name']}): attendance {rec['attendance_percent']}%, "
            f"capstone submitted: {'yes' if rec['capstone_submitted'] else 'no'}, "
            f"approved extension: {rec['extension_days']} days"
            + (f", so this student's capstone deadline is {14 + rec['extension_days']} December at 23:59."
               if rec["extension_days"] else ".")) if rec else "No such student."
    return {"evidence": [text], "sources": [f"student-system:{student_id}"],
            "strength": 1.0 if rec else 0.0}


def create_ticket(summary: str) -> dict:
    tickets = json.loads(TICKETS.read_text()) if TICKETS.exists() else []
    tid = f"T-{1000 + len(tickets) + 1}"
    tickets.append({"id": tid, "summary": summary})
    TICKETS.write_text(json.dumps(tickets, indent=1))
    return {"evidence": [f"Created ticket {tid}: {summary}"], "sources": [f"ticketing:{tid}"], "strength": 1.0}


TOOLS = {"search_course": "course rules: deadlines, grading, attendance, teams, search techniques",
         "search_company": "company policies: leave, travel, IT, laptops, payments, pricing",
         "student_record": "LIVE data for one student id like s-1002: attendance, submission, extensions",
         "create_ticket": "ACTION: open an IT/support ticket"}


# ------------------------------------------------------------------ router
def rule_router(q: str) -> list[str]:
    ql = q.lower()
    tools = []
    if re.search(r"\bs-\d{4}\b", ql):
        tools.append("student_record")
    if any(w in ql for w in ["ticket", "stolen", "broken"]):
        tools += ["search_company", "create_ticket"]
    elif any(w in ql for w in ["laptop", "leave", "expense", "vpn", "password"]):
        tools.append("search_company")
    if any(w in ql for w in ["quiz", "capstone", "late", "deadline", "grade", "team", "attendance"]):
        tools.append("search_course")
    return list(dict.fromkeys(tools))


def llm_router(q: str) -> list[str] | None:
    if not getattr(llm, "can_reason", False):
        return None
    menu = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
    out = llm.chat("You pick tools for an assistant. Reply with tool names separated by commas, or 'none'. "
                   "Use student_record whenever the user gives a student id. Use create_ticket only when "
                   "the user asks to open/create/report something.", f"Tools:\n{menu}\n\nRequest: {q}\nTools:", 30)
    names = [n for n in TOOLS if n in out]
    return names if names or "none" in out.lower() else None


def guardrails(q: str, tools: list[str]) -> list[str]:
    """Hard rules that hold whatever the router says."""
    if re.search(r"\bs-\d{4}\b", q.lower()) and "student_record" not in tools:
        tools = tools + ["student_record"]          # live data must come from the live system
    if re.search(r"\b(open|create|file|raise)\b.*\bticket\b", q.lower()) and "create_ticket" not in tools:
        tools = tools + ["create_ticket"]           # explicit action requests
    return tools


def choose_tools(q: str) -> tuple[list[str], str]:
    """RAG_ROUTER=llm | rules | auto. 'auto' trusts the LLM only when it is a capable model
    (Ollama or an OpenAI-compatible server); tiny local models route unreliably."""
    mode = os.getenv("RAG_ROUTER", "auto")
    capable = llm.name.startswith(("ollama", "openai"))
    suggestion = llm_router(q) if getattr(llm, "can_reason", False) else None
    if mode == "llm" or (mode == "auto" and capable and suggestion is not None):
        return guardrails(q, suggestion or []), f"LLM ({llm.name})"
    if suggestion is not None:
        show.info(f"   (the small LLM suggested {suggestion}; using rules instead, set RAG_ROUTER=llm to force it)")
    return guardrails(q, rule_router(q)), "rules"


def run_agent(q: str):
    show.step(f"User: {q}")
    tools, how = choose_tools(q)
    show.info(f"   Router ({how}) → {tools or 'no tool'}")
    if not tools:
        show.answer_box("I can only help with course and company questions, student records and IT tickets.")
        return
    observations = []
    for name in tools:
        arg = re.search(r"\bs-\d{4}\b", q.lower()).group(0) if name == "student_record" else q
        result = globals()[name](arg)
        if name.startswith("search") and result["strength"] < 0.40:      # self-check
            show.info(f"   {name}: evidence weak ({result['strength']:.2f}) → retry with rephrased queries")
            for alt in multi_query(q, llm)[1:3]:
                retry = globals()[name](alt)
                if retry["strength"] > result["strength"]:
                    result = retry
        show.info(f"   {name} → {result['sources']}")
        observations.append(result)
    context = [e for o in observations for e in o["evidence"]]
    from ragkit.llm import build_prompt
    answer_text = llm.chat("Answer the user using ONLY the numbered context. The context has rules from documents "
                           "and live records about this user: apply the rules to the records. Cite like [1]. "
                           "If the context is not enough, say you don't know.",
                           build_prompt(q, context), 200)
    show.answer_box(answer_text + "\n\nSources: " + ", ".join(s for o in observations for s in o["sources"]))


for q in ["What happens if I submit the capstone late?",
          "I'm student s-1002. Am I allowed to take the final quiz?",
          "I'm student s-1002 and haven't submitted the capstone yet. What is my deadline now?",
          "My laptop was stolen this morning, please open a ticket.",
          "What's the weather in Paris tomorrow?"]:
    run_agent(q)

show.takeaway("RAG supplies information; tools fetch live state or take actions.\n"
              "Agents add flexibility, plus latency, cost and new failure modes: evaluate the route too.")
