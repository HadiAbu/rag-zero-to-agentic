"""Pretty terminal output for the lessons (uses `rich` when installed)."""
from __future__ import annotations

import os

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=int(os.getenv("COLUMNS", "110")))
    RICH = True
except ImportError:  # plain-text fallback
    RICH = False
    console = None


def _p(text=""):
    if RICH:
        console.print(text)
    else:
        import re

        print(re.sub(r"\[/?[a-z ]+\]", "", str(text)))


def lesson(number: str, title: str, problem: str, idea: str):
    if RICH:
        console.rule(f"[bold]Lesson {number} · {title}")
        console.print(Panel(problem, title="❌ The problem", border_style="red"))
        console.print(Panel(idea, title="💡 The idea", border_style="cyan"))
    else:
        print("=" * 80, f"\nLesson {number} · {title}\nPROBLEM: {problem}\nIDEA: {idea}\n" + "=" * 80)


def step(title: str):
    if RICH:
        console.print(f"\n[bold yellow]▶ {title}[/bold yellow]")
    else:
        print(f"\n> {title}")


def info(text: str):
    _p(f"[dim]{text}[/dim]")


def good(text: str):
    _p(f"[bold green]✅ {text}[/bold green]")


def bad(text: str):
    _p(f"[bold red]❌ {text}[/bold red]")


def hits_table(hits, title: str = "Retrieved", expected=None, show_score: bool = True, width: int = 70, mark: str = "✅"):
    """expected: a function chunk -> bool that marks chunks (with `mark`, default ✅ = the right one)."""
    if RICH:
        t = Table(title=title, title_justify="left", show_lines=False)
        t.add_column("#", justify="right")
        t.add_column("source")
        if show_score:
            t.add_column("score", justify="right")
        t.add_column("text")
        for h in hits:
            flag = f" {mark}" if expected and expected(h.chunk) else ""
            text = h.chunk.text.replace("\n", " ")
            row = [str(h.rank), h.chunk.label + flag]
            if show_score:
                row.append(f"{h.score:.3f}")
            row.append(text[:width] + ("…" if len(text) > width else ""))
            t.add_row(*row)
        console.print(t)
    else:
        print(title)
        for h in hits:
            flag = f" <== {mark}" if expected and expected(h.chunk) else ""
            print(f"  {h.rank}. [{h.score:.3f}] {h.chunk.label}: {h.chunk.text[:width]}{flag}")


def answer_box(text: str, title: str = "Answer"):
    if RICH:
        console.print(Panel(text, title=title, border_style="green"))
    else:
        print(f"{title}: {text}")


def table(rows: list[dict], title: str = ""):
    if not rows:
        return
    if RICH:
        t = Table(title=title, title_justify="left")
        for k in rows[0]:
            t.add_column(str(k))
        for r in rows:
            t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()])
        console.print(t)
    else:
        print(title)
        for r in rows:
            print("  ", r)


def takeaway(text: str):
    if RICH:
        console.print(Panel(text, title="🎯 Takeaway", border_style="green"))
    else:
        print("TAKEAWAY:", text)


def setup_info(embedder=None, llm=None, reranker=None):
    parts = []
    if embedder is not None:
        parts.append(f"embedder={embedder.name}")
    if reranker is not None:
        parts.append(f"reranker={reranker.name}")
    if llm is not None:
        parts.append(f"llm={llm.name}")
    info("Using " + ", ".join(parts))
