"""
THE CHALLENGE · "The Knowledge Agent" (brief: lab/challenge/BRIEF.md)

Score YOUR pipeline (lab/rag_lab.py, with the steps and settings you chose) on hiring
questions about the candidate CVs, always asked by an interviewer (ROLE "employee").

    python lab/challenge.py                                  # 7 practice questions on the CVs
    python lab/challenge.py path/to/hidden.jsonl --team "Team 3"
    python lab/challenge.py --corpus company                 # the older company-documents set

Points per question
    exact / paraphrase / version   +2 the right source reaches the prompt, +1 more if it is #1
                                   -1 if an outdated version is also in the prompt
    restricted                     +3 nothing restricted reaches the prompt, -5 if it leaks
    not in docs                    +3 refused, -2 if passages were sent anyway
    spread-out                     +1 for every required item found
    match (a list of candidates)   +1 for every right candidate in the prompt, -1 for every wrong one
Ties: fewer passage tokens wins (a cheaper prompt is a better system).
"""
import argparse
import contextlib
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rag_lab as lab

HERE = pathlib.Path(__file__).resolve().parent


def score(item, result):
    hits = result["hits"]
    kind = item["type"]
    if kind == "restricted":
        leaked = any(h.chunk.meta.get("access") == "restricted" and lab.ROLE not in h.chunk.meta.get("allowed_roles", [])
                     for h in hits)
        return (-5, "LEAKED restricted text") if leaked else (3, "protected")
    if kind == "not in docs":
        return (3, "refused") if result["refused"] or not hits else (-2, "answered from unrelated text")
    if kind == "match":
        present = {h.chunk.meta.get("candidate_id") for h in hits} - {None}
        right, wrong = present & set(item["all"]), present - set(item["all"])
        return len(right) - len(wrong), f"{len(right)}/{len(item['all'])} right, {len(wrong)} wrong"
    if kind == "spread-out":
        text = " ".join(h.chunk.with_context() for h in hits)
        got = [x for x in item["all"] if x in text]
        return len(got), f"{len(got)}/{len(item['all'])} found"
    points, note = 0, "miss"
    for h in hits:
        if h.chunk.doc_id in item["docs"] and item.get("contains", "").lower() in h.chunk.text.lower():
            points, note = (3, "#1") if h.rank == 1 else (2, f"#{h.rank}")
            break
    if item.get("stale") and any(h.chunk.doc_id == item["stale"] for h in hits):
        points, note = points - 1, note + ", outdated version in prompt"
    return points, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default=None)
    ap.add_argument("--team", default="my team")
    ap.add_argument("--corpus", default="cvs", help="cvs (default) or company")
    args = ap.parse_args()
    args.questions = args.questions or str(HERE / "challenge" / ("cvs_practice.jsonl" if args.corpus == "cvs" else "practice.jsonl"))
    items = [json.loads(line) for line in pathlib.Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]

    lab.CORPUS, lab.ROLE, lab.GENERATE_ANSWERS = args.corpus, "employee", False
    with contextlib.redirect_stdout(io.StringIO()):
        lab.setup()
    from ragkit import show
    rows, total, tokens = [], 0, 0
    for item in items:
        with contextlib.redirect_stdout(io.StringIO()):
            result = lab.ask(item["question"], [])
        pts, note = score(item, result)
        total += pts
        tokens += result["tokens"]
        rows.append({"type": item["type"], "question": item["question"], "result": note, "points": pts})
    best = sum(len(i["all"]) if i["type"] in ("spread-out", "match") else 3 for i in items)
    show.table(rows, f"{args.team}")
    show.info(f"SCORE {total} / {best} · passage tokens {tokens} (tie-break: fewer wins)")


if __name__ == "__main__":
    main()
