"""
Before the session: is everything ready?    python lab/check_setup.py

Checks Python, the packages, the three models (downloads any that are missing, about 3.2 GB in
total, so run it at home), the data, and whether Claude Code or Kiro is installed.
Every line is ✓ or ✗ with what to do. Send the instructor a screenshot if anything stays ✗.
"""
import importlib
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OK, BAD = "✓", "✗"
problems = 0


def line(ok, what, fix=""):
    global problems
    problems += not ok
    print(f"  {OK if ok else BAD} {what}" + ("" if ok or not fix else f"\n      → {fix}"))


print("\n1 · Python and packages")
line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}", "install Python 3.10 or newer")
for mod, pip in [("numpy", "numpy"), ("sklearn", "scikit-learn"), ("rich", "rich"), ("sentence_transformers", "sentence-transformers"),
                 ("transformers", "transformers"), ("networkx", "networkx"), ("mcp", "\"mcp[cli]>=1.20,<2\""), ("gradio", "gradio")]:
    try:
        importlib.import_module(mod)
        line(True, mod)
    except Exception:
        line(False, mod, f"pip install {pip}   (inside your .venv)")

print("\n2 · Data")
for folder in ("course", "company", "cvs"):
    n = len(list((ROOT / "data" / folder).rglob("*.md")))
    line(n > 3, f"data/{folder}: {n} files", "git pull (the folder is missing)")

print("\n3 · Models (the first run downloads them)")
try:
    from ragkit import Reranker, get_embedder
    t = time.time(); e = get_embedder(); v = e.encode(["hello"])
    line(len(v[0]) > 10, f"embedder {e.name} ({time.time() - t:.0f} s)")
    t = time.time(); r = Reranker()
    line(True, f"reranker {r.name} ({time.time() - t:.0f} s)")
except Exception as ex:
    line(False, "embedder / reranker", f"{type(ex).__name__}: {ex}. Check the internet connection and run again.")
try:
    from huggingface_hub import snapshot_download
    name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    try:
        snapshot_download(name, local_files_only=True)
        line(True, f"LLM {name} is downloaded")
    except Exception:
        print(f"    downloading {name} (about 3 GB, once)…")
        snapshot_download(name)
        line(True, f"LLM {name} downloaded")
except Exception as ex:
    line(False, "LLM", f"{type(ex).__name__}: {ex}")

print("\n4 · Your AI assistant (you need one of them for step 11 and the challenge)")
found = [n for n in ("claude", "kiro", "kiro-cli") if shutil.which(n)]
line(bool(found), "Claude Code or Kiro: " + (", ".join(found) if found else "not found on PATH"),
     "install Claude Code (claude.com/claude-code) or Kiro (kiro.dev). Kiro IDE users can ignore this line.")

print("\n" + ("All set. See you at the session!" if not problems else f"{problems} thing(s) to fix above."))
