"""Smoke test: every lesson runs end to end in offline mode (no downloads)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LESSONS = sorted((ROOT / "lessons").glob("[0-9][0-9]_*.py"))
ENV = dict(os.environ, RAG_EMBEDDER="tfidf", RAG_LLM="extractive", RAG_RERANKER="overlap", COLUMNS="120")


@pytest.mark.parametrize("lesson", LESSONS, ids=[p.stem for p in LESSONS])
def test_lesson_runs_offline(lesson, tmp_path):
    result = subprocess.run([sys.executable, str(lesson)], env=ENV, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Takeaway" in result.stdout or "TAKEAWAY" in result.stdout
