"""Golden-set evaluation: did the expected source come back, and how high?"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .documents import DATA


def load_golden(name: str) -> list[dict]:
    path = DATA / "golden" / name
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk, item: dict) -> bool:
    if chunk.doc_id not in item.get("expected_docs", []):
        return False
    snippet = item.get("contains", "")
    return not snippet or snippet.lower() in chunk.text.lower()


def first_relevant_rank(hits, item) -> int | None:
    for h in hits:
        if is_relevant(h.chunk, item):
            return h.rank
    return None


def run_eval(retrieve, golden: list[dict], k: int = 3) -> dict:
    """retrieve(item) -> list[Hit]. Unanswerable items (no expected docs) are skipped here;
    refusal is checked separately in lesson 12."""
    rows = []
    for item in golden:
        if not item.get("expected_docs"):
            continue
        t0 = time.perf_counter()
        hits = retrieve(item)
        ms = (time.perf_counter() - t0) * 1000
        rank = first_relevant_rank(hits[:k], item)
        rows.append({"id": item["id"], "type": item.get("type", ""), "question": item["question"],
                     "hit": rank is not None, "rank": rank, "ms": ms})
    n = len(rows) or 1
    return {"rows": rows, "hit_rate": sum(r["hit"] for r in rows) / n,
            "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
            "avg_ms": sum(r["ms"] for r in rows) / n}
