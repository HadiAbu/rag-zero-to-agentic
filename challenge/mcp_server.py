"""
STEP 8 · Your CV chatbot as an MCP server (for Claude Code or Kiro)

It builds your pipeline exactly like challenge/app.py (load, chunk, index, graph) and exposes
two tools. The assistant's own model writes the answer; this server only finds passages.

  search_cvs(question)   -> the passages your retrieve() + should_refuse() return, with sources
  get_cv(file_name)      -> one whole CV, by the file name that search_cvs returned

Run it by hand to check it starts:  python challenge/mcp_server.py   (waits for a client; Ctrl+C to stop)
Connect it:                         the "cv-chatbot" entry in .mcp.json

Notes
- stdout is reserved for the MCP protocol, so every print from the pipeline goes to stderr.
- The role comes from the RAG_ROLE environment variable, never from a tool argument.
- The CVs are real people's data: they stay on this machine. The pipeline already redacts emails
  and phone numbers when it reads them (see _clean in pipeline.py).
- It rebuilds the Chroma index in challenge/chroma_db, like the app does: do not run the app
  and this server at the same time.
"""
import contextlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mcp.server.fastmcp import FastMCP

import pipeline

CV_DIR = pathlib.Path(os.getenv("CV_DIR", HERE / "cvs"))
ROLE = os.getenv("RAG_ROLE", "employee")          # read from the environment, never from the model

DOCS = {}                                          # file name -> text

with contextlib.redirect_stdout(sys.stderr):       # stdout is the protocol channel
    _docs = pipeline.load_cvs(CV_DIR)
    if not _docs:
        raise SystemExit(f"no CVs found in {CV_DIR}")
    _chunks = [c for d in _docs for c in pipeline.chunk(d)]
    pipeline.build_index(_chunks)
    try:
        pipeline.build_graph(_chunks)
    except Exception as e:                         # the graph is optional (STEP 7)
        print("build_graph failed:", e)
    DOCS.update({d["source"]: d["text"] for d in _docs})

mcp = FastMCP("cv-chatbot")


@mcp.tool()
def search_cvs(question: str) -> dict:
    """Search the candidate CVs. Returns the passages to answer from, each with its source file.
    Ask one focused question per call. If 'refused' is true, the CVs do not contain the answer:
    say so instead of guessing. Cite the source file for every fact."""
    with contextlib.redirect_stdout(sys.stderr):
        hits = pipeline.retrieve(question, pipeline.K)
        refused = bool(pipeline.should_refuse(question, hits))
    return {
        "question": question,
        "refused": refused,
        "passages": [] if refused else [
            {"source": h["source"], "text": h["text"], "score": round(float(h["score"]), 3), "found_by": h.get("via", "")}
            for h in hits
        ],
    }


@mcp.tool()
def get_cv(file_name: str) -> dict:
    """Return one whole CV by its file name (for example 'Jane_Doe_CV.pdf'), as listed in the
    'source' field of search_cvs results."""
    if file_name in DOCS:
        return {"source": file_name, "text": DOCS[file_name]}
    return {"error": f"No CV named '{file_name}'.", "known_files": sorted(DOCS)}


if __name__ == "__main__":
    mcp.run()
