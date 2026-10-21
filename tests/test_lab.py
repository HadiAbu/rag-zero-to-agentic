"""Switch on the lab's steps one by one, exactly as a student would, and run each version."""
import os
import pathlib
import re
import subprocess
import sys

import pytest

LAB = pathlib.Path(__file__).resolve().parents[1] / "lab" / "rag_lab.py"


def switch_on(source: str, up_to: int) -> str:
    """Uncomment STEP 2..up_to the way Ctrl+/ does: '## x' -> '# x', '# x' -> 'x', '#' -> ''."""
    out, step = [], None
    for line in source.split("\n"):
        m = re.match(r"## STEP (\d+) ", line)
        if m:
            step = int(m.group(1))
        if line.startswith("# ---------------------------------------------------------------- the test questions"):
            step = None
        if step and 2 <= step <= up_to and line.startswith("#"):
            line = line[1:] if line.startswith("##") else line[2:] if line.startswith("# ") else line[1:]
        out.append(line)
    return "\n".join(out)


def run_steps(up_to: int, tmp_path) -> str:
    path = tmp_path / "rag_lab.py"
    path.write_text(switch_on(LAB.read_text(), up_to))
    env = dict(os.environ, RAG_EMBEDDER=os.getenv("RAG_EMBEDDER", "tfidf"), RAG_LLM=os.getenv("RAG_LLM", "extractive"),
               RAG_RERANKER=os.getenv("RAG_RERANKER", "overlap"), COLUMNS="140",
               PYTHONPATH=str(LAB.parents[1]))
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env, cwd=LAB.parents[1],
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.mark.parametrize("up_to", range(1, 11))
def test_every_step_runs(up_to, tmp_path):
    assert "Scoreboard" in run_steps(up_to, tmp_path)
