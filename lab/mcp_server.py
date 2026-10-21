"""
STEP 11 · Your RAG pipeline as an MCP server (for Claude Code, Claude Desktop or Kiro)

It imports lab/rag_lab.py, so it serves exactly the steps you switched on there.
The assistant's own model writes the answer; this server only finds passages.

Run it by hand to check it starts:   python lab/mcp_server.py   (it waits for a client; Ctrl+C to stop)
Connect it:                          see lab/README.md, "Connect the MCP server"

Security: the user's role comes from the RAG_ROLE environment variable, set by whoever
starts the server. It is NOT a tool argument, so the model cannot give itself more rights.
"""
import contextlib
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

import rag_lab as lab

lab.CORPUS = os.getenv("RAG_CORPUS", "company")
lab.ROLE = os.getenv("RAG_ROLE", "employee")
lab.GENERATE_ANSWERS = False             # the assistant writes the answer, not our local model

with contextlib.redirect_stdout(sys.stderr):   # stdout is reserved for the MCP protocol
    lab.setup()

mcp = FastMCP("rag-lab")


@mcp.tool()
def search_docs(question: str) -> dict:
    """Search the company knowledge base. Returns the passages to answer from, each with its
    source. If 'refused' is true, the documents do not contain the answer: say so."""
    with contextlib.redirect_stdout(sys.stderr):
        result = lab.ask(question, [])
    return {
        "searched_for": result["query"],
        "refused": result["refused"],
        "passages": [{"source": h.chunk.label, "text": text}
                     for h, text in zip(result["hits"], result["texts"])],
    }


@mcp.tool()
def get_document(doc_id: str) -> dict:
    """Return one whole document by its id (for example 'leave-policy-2026'), if the user may see it."""
    for d in lab.DOCS:
        if d.doc_id == doc_id:
            if d.meta.get("access") == "restricted" and lab.ROLE not in d.meta.get("allowed_roles", []):
                return {"error": "You do not have access to this document."}
            return {"doc_id": d.doc_id, "title": d.title, "status": d.meta.get("status", "current"), "text": d.text}
    return {"error": f"No document '{doc_id}'. Known ids: " + ", ".join(sorted(d.doc_id for d in lab.DOCS))}


@mcp.tool()
def graph_neighbors(entity: str) -> dict:
    """List what is connected to an entity such as ERR-5007 or POL-204 in the knowledge graph (STEP 10)."""
    if lab.GRAPH is None:
        return {"error": "The graph is off. Switch on STEP 10 in lab/rag_lab.py and restart the server."}
    node = ("entity", entity.upper())
    if node not in lab.GRAPH:
        return {"error": f"'{entity}' is not in the graph."}
    out = {"entity": entity.upper(), "attributes": dict(lab.GRAPH.nodes[node]), "related": [], "mentioned_in": []}
    for n in lab.GRAPH.neighbors(node):
        if n[0] == "entity":
            out["related"].append(n[1])
        elif lab.FILTER is None or lab.FILTER(lab.CHUNKS[n[1]]):
            out["mentioned_in"].append(lab.CHUNKS[n[1]].label)
    return out


if __name__ == "__main__":
    mcp.run()   # stdio: the assistant starts this file as a child process
