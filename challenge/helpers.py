"""Given to you: the free local models. You don't need to change this file.

    embed(["text one", "text two"])  ->  numpy array, one normalized vector per text (MiniLM, 384 numbers)
    llm(system, user)                ->  the model's answer as a string (Qwen2.5-1.5B on your laptop)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ragkit import get_embedder, get_llm   # noqa: E402

_EMBEDDER, _LLM = None, None


def embed(texts):
    """Turn texts into vectors. Same model for documents and questions."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    v = np.asarray(_EMBEDDER.encode(list(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def llm(system, user, max_tokens=300):
    """Ask the LLM. Loads the model the first time (about 10 seconds)."""
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM.chat(system, user, max_tokens)


def llm_name():
    return getattr(_LLM, "name", "not loaded yet")
