"""
YOUR PIPELINE · the only file you edit today
=============================================
The app (challenge/app.py) calls these functions. You build them level by level, with Claude Code
or Kiro, and press "Reload pipeline" in the app after every change. The status box at the top of
the app shows which levels work.

  LEVEL 1 · Ingest   load_cvs, chunk          → the 📄 Data tab shows your CVs and chunks
  LEVEL 2 · Index    build_index (Chroma)     → the status shows how many chunks are stored
  LEVEL 3 · Answer   retrieve, build_prompt   → the 💬 Chat works: this is NAIVE RAG.
                                                Run ✅ Verify: that is your first score.
  LEVEL 4 · Features make_query, retrieve, should_refuse, build_graph … (see challenge/README.md)

Given to you (helpers.py):  embed(texts) -> normalized vectors · llm(system, user) -> answer text
Stuck or behind?  cp challenge/checkpoints/level3.py challenge/pipeline.py   (naive RAG, done)
"""
import pathlib
import re

import datetime
import difflib
import logging
import math
import time
import unicodedata
from collections import Counter

import numpy as np

import chromadb
import docx
import networkx as nx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

try:
    import pymupdf                             # PDF reader: correct reading order, tables (AGPL licence)
except ImportError:
    pymupdf = None

from helpers import embed
from ragkit import Reranker                       # helpers.py puts the repo root on sys.path

logging.getLogger("pypdf").setLevel(logging.ERROR)    # silence "wrong pointing object" noise

HERE = pathlib.Path(__file__).resolve().parent
K = 3                     # chunks per prompt
REFUSAL = "I don't know based on the CVs."
SYSTEM_PROMPT = ("You are a recruiting assistant. Answer ONLY from the numbered CV passages. "
                 "Cite every fact like [1]. If the passages do not contain the answer, reply exactly: "
                 f"\"{REFUSAL}\" Be brief.")


# ================================================================= LEVEL 1 · INGEST
# Prompt idea: "In challenge/pipeline.py implement load_cvs and chunk as their docstrings say.
#  Use pypdf for PDF and python-docx for DOCX. Start with fixed-size chunks of 800 characters
#  with 100 overlap. Don't change other files."
CHUNK_SIZE, OVERLAP = 800, 100
PREFIX_NAME = False           # METADATA experiment: start every chunk with "CV: <name from the file name>".
                              # Measured: name questions 96% -> 100%, but it shifts the scores enough to break the
                              # refusal rule (p09 no longer refused). Kept off; use the name as a FILTER instead.


def _name_from_file(source):
    """A readable name from the file name: 'Ajwad_Zidan_CV.pdf' -> 'Ajwad Zidan' (also strips a glued 'CV')."""
    stem = re.sub(r"(?i)(cv|resume)(\W|\d|$)", " ", pathlib.Path(source).stem.replace(".docx", ""))
    return " ".join(stem.replace("_", " ").replace("-", " ").split()).title()
_SPACED = re.compile(r"\b(?:\w ){3,}\w\b")        # "P R O F I L E": 4+ single characters


def _unspace(text):
    """Join letter-spaced words ("P R O F I L E" -> "PROFILE"); ordinary text is left alone."""
    return _SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{7,}\d(?!\w)")
_YEAR = re.compile(r"(?:19|20)\d\d")
_PAGE_LINE = re.compile(r"(?:page\s*)?\d{1,2}(?:\s*(?:of|/)\s*\d{1,2})?", re.I)


def _redact_phone(m):
    """Replace a phone number, but leave date ranges like '2019 - 2023' alone."""
    s = m.group(0)
    digits = sum(ch.isdigit() for ch in s)
    if _YEAR.search(s) and not s.lstrip().startswith("+") or not 9 <= digits <= 15:
        return s
    return "[phone]"


def _clean(text):
    """Remove the noise that carries no meaning, and keep contact details out of the index."""
    text = unicodedata.normalize("NFKC", text)                      # "ﬁ" -> "fi"
    text = "".join(" " if unicodedata.category(c) in ("So", "Co", "Cf") or (unicodedata.category(c) == "Cc" and c not in "\n\t")
                   else c for c in text)                            # bullets, icons, DEL, zero-width marks
    text = "\n".join(l for l in text.split("\n") if not _PAGE_LINE.fullmatch(l.strip()))
    text = _PHONE.sub(_redact_phone, _EMAIL.sub("[email]", text))
    return re.sub(r"[ \t]{2,}", " ", text)


def _blocks(parent_elm, parent):
    """Yield the text lines of a document body or table cell in reading order,
    so paragraphs and tables stay interleaved (and tables may be nested)."""
    for el in parent_elm.iterchildren():
        if el.tag == qn("w:p"):
            yield Paragraph(el, parent).text
        elif el.tag == qn("w:tbl"):
            yield from _table_lines(Table(el, parent))


def _table_lines(table):
    for row in table.rows:
        cells, seen = [], []
        for c in row.cells:
            if any(c._tc is s for s in seen):      # a merged cell repeats once per column it spans
                continue
            seen.append(c._tc)
            text = "\n".join(l for l in _blocks(c._tc, c) if l.strip())
            if text:
                cells.append(text)
        if all("\n" not in c and len(c) < 80 for c in cells):
            yield " | ".join(cells)                # a data row: skill | level
        else:
            yield from cells                       # a layout cell holding whole sections


def _read_docx(path):
    d = docx.Document(str(path))
    return "\n".join(_blocks(d.element.body, d))


def _mostly_one_word_lines(text):
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines) > 20 and sum(len(l.split()) <= 1 for l in lines) / len(lines) > 0.5


def _pdf_page(page):
    """One page in reading order, with detected tables written as "cell | cell" rows (like the DOCX reader)."""
    tables = page.find_tables().tables
    boxes = [pymupdf.Rect(t.bbox) for t in tables]
    items = []
    for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", sort=True):
        r = pymupdf.Rect(x0, y0, x1, y1)
        inside = any((r & b).get_area() > 0.5 * r.get_area() for b in boxes if r.get_area() > 0)
        if kind == 0 and text.strip() and not inside:    # kind 0 = text; image blocks are skipped
            items.append((y0, x0, text.strip()))
    for t, b in zip(tables, boxes):
        rows = [" | ".join(" ".join(c.split()) for c in row if c and c.strip()) for row in t.extract()]
        items.append((b.y0, b.x0, "\n".join(r for r in rows if r)))
    return "\n".join(text for _, _, text in sorted(items))


def _read_pdf(path):
    if pymupdf is not None:
        with pymupdf.open(str(path)) as doc:
            return "\n".join(_pdf_page(page) for page in doc)
    return _read_pdf_pypdf(path)                   # PyMuPDF not installed


def _read_pdf_pypdf(path):
    pages = PdfReader(str(path)).pages
    text = "\n".join(page.extract_text() or "" for page in pages)
    if _mostly_one_word_lines(text):               # some PDFs come out as one word per line; layout mode rebuilds the lines
        text = "\n".join(page.extract_text(extraction_mode="layout") or "" for page in pages)
    return text


def _read_file(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".docx":
        text = _read_docx(path)
    elif ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        return None
    text = _clean(_unspace(text))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_cvs(folder):
    """Read every CV in `folder` (PDF, DOCX, TXT, MD).
    Return a list of documents: [{"source": "<file name>", "text": "<all the text>"}, ...]
    `source` must be the FILE NAME: Verify checks retrieval against it."""
    docs = []
    for path in sorted(pathlib.Path(folder).iterdir()):
        if path.name.startswith(("~$", ".")):      # Word lock files, hidden files
            continue
        try:
            text = _read_file(path)
        except Exception as e:                     # keep the CV so the Data tab can warn about it
            print(f"could not read {path.name}: {type(e).__name__}: {e}")
            text = ""
        if text is not None:
            docs.append({"source": path.name, "text": text})
    return docs


_HEADINGS = [("summary", r"(?:professional )?summary|profile|objective|about(?: me)?"),
             ("experience", r"(?:work |professional |employment |relevant )?(?:experience|history)|employment"),
             ("education", r"education|academic.*"),
             ("skills", r"(?:technical |core )?skills"),
             ("projects", r"projects?"),
             ("languages", r"languages"),
             ("certifications", r"certifications?|courses"),
             ("military", r"military.*"),
             ("volunteering", r"volunteer.*")]
_HEADING_RES = [(name, re.compile(rf"\s*(?:{pat})\s*:?\s*", re.I)) for name, pat in _HEADINGS]


def _sections(text):
    """[(char offset, section name)] for every heading line, in order. A heading is a short line."""
    found, pos = [], 0
    for line in text.split("\n"):
        if 0 < len(line.split()) <= 4:
            for name, rx in _HEADING_RES:
                if rx.fullmatch(line):
                    found.append((pos, name))
                    break
        pos += len(line) + 1
    return found


def _section_at(sections, offset):
    return next((name for pos, name in reversed(sections) if pos <= offset), "header")


def _sections_covered(sections, text_len, start, end, min_chars=150):
    """Every section that fills at least `min_chars` of the window [start, end): an 800-character chunk often
    starts in one section and holds most of the next, so the section it starts in is not enough."""
    bounds = [(0, "header")] + list(sections) + [(text_len, None)]
    found = []
    for (pos, name), (nxt, _) in zip(bounds, bounds[1:]):
        if name and min(end, nxt) - max(start, pos) >= min_chars and name not in found:
            found.append(name)
    return found or [_section_at(sections, start)]


_MONTH = r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*|\d{1,2}/)?"
_RANGE = re.compile(rf"{_MONTH}((?:19|20)\d\d)\s*[-–—]+\s*(?:{_MONTH}((?:19|20)\d\d)|(present|now|current|ongoing))", re.I)
_EXPLICIT_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs)\s*(?:of\s*)?(?:[a-z-]+\s+){0,3}experience", re.I)


def _years_of_experience(text, sections):
    """Estimated years of work experience: the date ranges inside the Experience section (every range outside
    Education when the CV has no such heading), overlaps merged, "Present" counted up to this year. A stated
    "5+ years of experience" counts too; the larger number wins. An estimate, accurate to about a year."""
    this_year = datetime.date.today().year
    spans = [(pos, name) for pos, name in sections]
    has_exp = any(name == "experience" for _, name in spans)
    ranges = []
    for m in _RANGE.finditer(text):
        sec = _section_at(sections, m.start())
        if (sec == "experience") if has_exp else (sec not in ("education", "projects", "certifications", "military")):
            start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else this_year
            if 1980 <= start <= end <= this_year + 1:
                ranges.append((start, min(end, this_year)))
    total, last_end = 0, None
    for start, end in sorted(ranges):              # merge overlapping jobs so parallel roles are not counted twice
        if last_end is None or start > last_end:
            total += end - start
            last_end = end
        elif end > last_end:
            total += end - last_end
            last_end = end
    stated = max((int(n) for n in _EXPLICIT_YEARS.findall(text)), default=0)
    return max(total, stated)


def chunk(doc):
    """Split ONE document into chunks.
    Return [{"id": "<unique id>", "source": doc["source"], "text": "<chunk text>"}, ...]
    You may add more keys (they become metadata), e.g. "name", "section", "city"."""
    text, chunks = doc["text"], []
    name = _name_from_file(doc["source"])
    sections = _sections(text)
    years = _years_of_experience(text, sections)
    for i, start in enumerate(range(0, len(text), CHUNK_SIZE - OVERLAP)):
        piece = text[start:start + CHUNK_SIZE].strip()
        if piece:
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "name": name,
                           "section": _section_at(sections, start), "years": years,
                           "sections": "|".join(_sections_covered(sections, len(text), start, start + CHUNK_SIZE)),
                           "text": f"CV: {name}\n{piece}" if PREFIX_NAME else piece})
        if start + CHUNK_SIZE >= len(text):        # the rest is already inside this chunk
            break
    return chunks


# ================================================================= LEVEL 2 · INDEX (the vector DB)
# Prompt idea: "Implement build_index with chromadb: a PersistentClient in HERE / 'chroma_db',
#  delete and recreate a collection 'cvs' with cosine distance, add every chunk with its id,
#  text, metadata and embed(texts) as embeddings. Return the collection count."
COLLECTION = None
NAME_WORDS = {}          # METADATA FILTER: each word of a candidate's name -> the CV files it belongs to
SOURCE_CHUNKS = {}       # CV file -> how many chunks it has
COMMON_NAME_WORDS = set()   # name words that are also ordinary words inside the CVs ("ward"): must be capitalised


def _index_names(chunks):
    NAME_WORDS.clear(), SOURCE_CHUNKS.clear(), COMMON_NAME_WORDS.clear()
    for c in chunks:
        SOURCE_CHUNKS[c["source"]] = SOURCE_CHUNKS.get(c["source"], 0) + 1
        for w in re.findall(r"[A-Za-z]{3,}", c.get("name", "")):
            NAME_WORDS.setdefault(w.lower(), set()).add(c["source"])
    body = " ".join(c["text"] for c in chunks)
    for w in NAME_WORDS:                           # lowercase occurrences only: the pattern is case-sensitive
        if len(re.findall(rf"(?<![A-Za-z]){w}(?![A-Za-z])", body)) >= 2:
            COMMON_NAME_WORDS.add(w)
    COMMON_NAME_WORDS.update(_ordinary_words(NAME_WORDS))


def _ordinary_words(words):
    """Name words that the embedding model's own vocabulary knows as ordinary English ("ward", "hope"):
    they only count as a name when the question capitalises them."""
    try:
        import helpers
        embed(["load the model"])                  # makes sure the embedder (and its tokenizer) is loaded
        vocab = helpers._EMBEDDER.model.tokenizer.get_vocab()
        return {w for w in words if w in vocab}
    except Exception:                              # another embedder: fall back to the corpus counts only
        return set()


def _named_sources(query):
    """The CV files whose candidate is named in the question (empty when nobody is). A name word votes for
    its CVs; only the CVs with the most votes stay, so "Tasneem Youssef" picks one person, not every Youssef."""
    votes = {}
    long_words = [n for n in NAME_WORDS if len(n) >= 5]
    for word in re.findall(r"[A-Za-z]{3,}", query):
        w = word.lower()
        if w not in NAME_WORDS:                    # a spelling variant: Ibrahim / Ibrahem
            # 0.84: measured on the CVs' own vocabulary; at 0.80 ordinary words like "based" match a name
            close = difflib.get_close_matches(w, long_words, n=1, cutoff=0.84) if len(w) >= 5 else []
            if not close:
                continue
            w = close[0]
        if w in COMMON_NAME_WORDS and not word[0].isupper():
            continue
        for src in NAME_WORDS[w]:
            votes[src] = votes.get(src, 0) + 1
    top = max(votes.values(), default=0)
    return sorted(s for s, v in votes.items() if v == top)


def build_index(chunks):
    """Store every chunk with its vector in a Chroma collection. Return how many are stored."""
    global COLLECTION
    client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))
    try:
        client.delete_collection("cvs")            # rebuild from scratch: no stale chunks after a Reload
    except Exception:
        pass
    COLLECTION = client.create_collection("cvs", metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 64):            # embed in batches to keep memory small
        batch = chunks[i:i + 64]
        COLLECTION.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embed([c["text"] for c in batch]).tolist(),
            metadatas=[{k: v for k, v in c.items() if k not in ("id", "text")} for c in batch],
        )
    _index_names(chunks)
    _index_keywords(chunks)                        # STEP 3: the BM25 side of the hybrid search
    return COLLECTION.count()


# ================================================================= LEVEL 3 · RETRIEVE AND PROMPT
# Prompt idea: "Implement retrieve: embed the query, query the Chroma collection for k results,
#  and return them as dicts with rank, source, text, score (1 - distance) and via='vector'.
#  Implement build_prompt: number the passages [1], [2]… with their source, then the question."
def retrieve(query, k=K):
    """Return the k best chunks: [{"rank": 1, "source": ..., "text": ..., "score": ..., "via": "vector"}, ...]"""
    named = _named_sources(query)                  # METADATA: a question about one named person searches only their CV
    if named:
        return _semantic(query, k, named)
    graph_hits = _graph_search(query)              # STEP 7: "which candidates know X?" is answered by the graph
    if graph_hits is None:
        return _semantic(query, k)
    if graph_hits and len(graph_hits) < k:         # the graph can be incomplete: let the vector search back it up
        have = {h["source"] for h in graph_hits}
        graph_hits += [h for h in _semantic(query, k + len(have)) if h["source"] not in have][:k - len(graph_hits)]
        graph_hits = [{**h, "rank": i} for i, h in enumerate(graph_hits, 1)]
    return graph_hits


def _semantic(query, k, only=None):
    """Vector search over Chroma, then the cross-encoder reranks a wider pool (STEP 4).
    `only`: a list of CV files to search in (the metadata filter), or None for all CVs."""
    hits = _vector_search(query, CANDIDATES if (RERANK or HYBRID) else k, only)
    if HYBRID:                                     # STEP 3: the keyword list joins the vector list before the rerank
        hits = _fuse(hits, _keyword_search(query, CANDIDATES, only), query)[:CANDIDATES]
    tag = " · name filter" if only else ""
    if not RERANK:
        return [{**h, "via": h["via"] + tag, "rank": i} for i, h in enumerate(hits[:k], 1)]
    t0 = time.perf_counter()
    scores = _reranker().score(query, [h["text"] for h in hits])
    wanted = _wanted_sections(query)               # the raw score stays as it is (the refusal rule reads it)
    ranked = sorted(zip(scores, hits), key=lambda p: -(p[0] + (SECTION_BONUS if wanted & set(p[1]["sections"].split("|")) else 0)))[:k]
    print(f"rerank: {len(hits)} candidates in {(time.perf_counter() - t0) * 1000:.0f} ms")
    return [{**h, "rank": i, "vector_score": h["score"], "score": s, "via": h["via"] + "+rerank" + tag}
            for i, (s, h) in enumerate(ranked, 1)]


def _vector_search(query, n, only=None):
    if COLLECTION is None:
        raise RuntimeError("the index is not built yet: press Reload pipeline")
    where = None if not only else ({"source": only[0]} if len(only) == 1 else {"source": {"$in": list(only)}})
    pool = sum(SOURCE_CHUNKS.get(s, 0) for s in only) if only else COLLECTION.count()
    res = COLLECTION.query(query_embeddings=embed([query]).tolist(), n_results=max(1, min(n, pool)), where=where,
                           include=["documents", "metadatas", "distances"])
    return [{"rank": i, "id": cid, "source": meta["source"], "text": text, "score": 1 - dist, "via": "vector",
             "section": meta.get("section", ""), "sections": meta.get("sections", "")}
            for i, (cid, text, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                                            res["distances"][0]), 1)]


# METADATA · section routing: the wording of a question often says which part of a CV holds the answer
SECTION_BONUS = 3.0                            # added to the reranker score for ordering only; 0 switches it off
_SECTION_INTENT = [
    ("education", re.compile(r"\b(study|studied|studies|degrees?|universit\w+|college|graduat\w+|bachelor\w*|master\w*|"
                             r"b\.?sc|m\.?sc|gpa|school|education|alma mater|majored?)\b", re.I)),
    ("skills", re.compile(r"\b(skills?|tech stack|technologies|tools|proficien\w+|programming languages?)\b", re.I)),
    ("languages", re.compile(r"(?<!programming )\b(languages?|speaks?|spoken|fluent|fluency)\b", re.I)),
    ("certifications", re.compile(r"\b(certificat\w+|certified|courses?)\b", re.I)),
]


def _wanted_sections(query):
    """The CV sections the question points at (empty for the broad 'what did he do' questions)."""
    return {name for name, rx in _SECTION_INTENT if rx.search(query)}


# ================================================================= STEP 3 · HYBRID SEARCH (BM25 + RRF)
# Vector search finds PARAPHRASES but blurs rare exact terms (a tool, a company, a certificate). BM25 counts
# words, so it nails those but misses a paraphrase. Their scores live on different scales (cosine 0..1 against
# BM25 0..20), so they are merged by RANK: reciprocal rank fusion, score = sum of 1 / (60 + rank) over the lists.
HYBRID = True                                  # set False to compare in Verify
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75
BM25 = {}                                      # filled by build_index
_STOPWORDS = set("a about an and are as at be but by can do does for from has have how i if in is it its me my of on or "
                 "our should so that the their them then there these they this to us was we what when where which who "
                 "why will with would you your".split())


def _tokens(text):
    """Lowercase words. Keeps "node.js", "c++" and "c#" whole, which a plain \\w+ tokenizer would break."""
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*(?:\+\+|#)?", text.lower()) if t not in _STOPWORDS]


def _index_keywords(chunks):
    docs = [_tokens(c["text"]) for c in chunks]
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    BM25.clear()
    BM25.update(chunks=chunks, tf=[Counter(d) for d in docs], length=np.array([len(d) for d in docs], dtype=float),
                avg=float(np.mean([len(d) for d in docs])) if docs else 1.0,
                idf={t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()})   # rare word = big weight


def _keyword_search(query, n, only=None):
    """BM25: the rarer a query word is in the CVs and the more often a chunk uses it, the higher the score."""
    q = [t for t in _tokens(query) if t in BM25["idf"]]
    scored = []
    for i, tf in enumerate(BM25["tf"]):
        c = BM25["chunks"][i]
        if only and c["source"] not in only:
            continue
        norm = BM25_K1 * (1 - BM25_B + BM25_B * BM25["length"][i] / BM25["avg"])
        s = sum(BM25["idf"][t] * tf[t] * (BM25_K1 + 1) / (tf[t] + norm) for t in q if t in tf)
        if s > 0:
            scored.append((s, i))
    return [{"rank": r, "id": BM25["chunks"][i]["id"], "source": BM25["chunks"][i]["source"],
             "text": BM25["chunks"][i]["text"], "score": s, "via": "keyword",
             "section": BM25["chunks"][i].get("section", ""), "sections": BM25["chunks"][i].get("sections", "")}
            for r, (s, i) in enumerate(sorted(scored, reverse=True)[:n], 1)]


def _fuse(vector_hits, keyword_hits, query):
    """Reciprocal rank fusion of the two ranked lists. Each chunk keeps its VECTOR similarity as `score`, because
    the refusal rule reads it: a chunk that only BM25 found gets its cosine from the stored embedding."""
    entries = {}
    for tag, hits in (("v", vector_hits), ("k", keyword_hits)):    # the vector list goes first, so its hit is kept
        for h in hits:
            entries.setdefault(h["id"], {"hit": h, "ranks": {}})["ranks"][tag] = h["rank"]
    fused, keyword_only = [], []
    for cid, e in entries.items():
        found = ",".join(f"{t}{r}" for t, r in sorted(e["ranks"].items(), reverse=True))
        fused.append({**e["hit"], "rrf": sum(1 / (RRF_K + r) for r in e["ranks"].values()), "via": f"hybrid[{found}]"})
        if "v" not in e["ranks"]:
            keyword_only.append(cid)
    if keyword_only:
        embs = COLLECTION.get(ids=keyword_only, include=["embeddings"])
        qvec = embed([query])[0]
        cos = {i: float(np.dot(v, qvec)) for i, v in zip(embs["ids"], embs["embeddings"])}
        for h in fused:
            if h["id"] in cos:
                h["score"] = cos[h["id"]]
    fused.sort(key=lambda h: -h["rrf"])
    return [{**h, "rank": r} for r, h in enumerate(fused, 1)]


RERANK, CANDIDATES = True, 20                  # STEP 4: set RERANK = False to compare in Verify
_RERANKER = None


def _reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()                 # cross-encoder/ms-marco-MiniLM-L-6-v2, about 90 MB
    return _RERANKER


def build_prompt(question, hits):
    """The text the LLM receives: the numbered passages, then the question."""
    graph = [h for h in hits if h.get("via", "").startswith("graph")]
    if graph:                                      # a list question: the graph already holds the answer, so the small
        answer = "\n".join(f"{h['name']} [{h['rank']}]" for h in graph)     # LLM only has to COPY it (copying is
        return (f"Question: {question}\n"                                    # far more reliable than selecting)
                f"Copy this answer exactly, without changing, adding or removing anything:\n{answer}")
    passages = "\n\n".join(f"[{h['rank']}] ({h['source']})\n{h['text']}" for h in hits)
    return f"CV passages:\n\n{passages}\n\nQuestion: {question}"


# ================================================================= LEVEL 4 · FEATURES
# These already work in their simplest form. Each step of the menu replaces one of them
# (or retrieve / chunk above). One step at a time, then Verify.
_REFERS_BACK = re.compile(r"\b(he|she|him|her|his|hers|they|them|same)\b", re.I)   # not "their": "workers wear their…"
_OPENER = re.compile(r"\s*(and|also|plus|what about|how about|then)\b", re.I)


def _is_follow_up(question):
    """A question that cannot be searched alone: it points back at a person or opens with "And…"."""
    return bool(_REFERS_BACK.search(question) or _OPENER.match(question))


def make_query(question, history):
    """STEP 1 · Follow-ups: rewrite "and where did he study?" using `history` (earlier questions).
    Rule: add the most recent standalone question, so a chain of follow-ups keeps its subject."""
    if not history or not _is_follow_up(question):
        return question
    base = next((h for h in reversed(history) if not _is_follow_up(h)), history[0])
    return f"{base} {question}"


REFUSE_RERANK, REFUSE_VECTOR = -6.0, 0.36      # measured on the practice questions + 11 of my own


def should_refuse(query, hits):
    """STEP 6 · Return True when nothing relevant came back: the app answers REFUSAL without the LLM.
    Refuse only when BOTH signals are weak for the chunk we would trust most (the reranker's top one): its
    reranker score is very low AND its vector similarity is low. A paraphrase can fool one, rarely both.
    (The grounded SYSTEM_PROMPT is the second line of defence for the borderline cases.)"""
    if not hits:
        return True
    if hits[0].get("via", "").startswith("graph"):  # an exact connection in the graph is evidence, not a guess
        return False
    if "vector_score" not in hits[0]:              # RERANK is off: the vector score alone cannot separate them
        return False
    return hits[0]["score"] < REFUSE_RERANK and hits[0]["vector_score"] < REFUSE_VECTOR


GRAPH = nx.Graph()
MAX_GRAPH_HITS = 12                            # keep the prompt small for the local 1.5B model

# 7a · the vocabulary: general tech skills and cities (not taken from any question)
SKILLS = """python java javascript typescript c++ c# .net php ruby rust kotlin scala matlab golang r-studio bash
sql nosql mysql postgresql oracle mongodb redis elasticsearch cassandra dynamodb sqlite snowflake bigquery
kafka rabbitmq spark hadoop airflow dbt etl power-bi tableau looker qlik excel pandas numpy scikit-learn
tensorflow pytorch keras opencv nlp llm machine-learning deep-learning computer-vision data-science
react angular vue svelte next.js node.js express.js django flask fastapi spring-boot laravel graphql rest
html css sass tailwind bootstrap webpack redux
docker kubernetes terraform ansible jenkins github-actions gitlab ci/cd aws azure gcp linux nginx git
selenium cypress playwright pytest junit jmeter postman appium cucumber robot-framework
jira confluence figma photoshop agile scrum kanban salesforce sap n8n zapier power-automate
autocad solidworks labview plc vhdl verilog fpga arduino raspberry-pi""".split()
AMBIGUOUS = {"excel"}                          # also an ordinary verb: only the capitalised "Excel" counts
CITIES = ["tel aviv", "jerusalem", "haifa", "nazareth", "beer sheva", "be'er sheva", "netanya", "herzliya",
          "ramat gan", "petah tikva", "petach tikva", "rishon lezion", "holon", "bat yam", "ashdod", "ashkelon",
          "rehovot", "kfar saba", "ra'anana", "raanana", "hod hasharon", "modiin", "afula", "tiberias", "akko",
          "acre", "nahariya", "karmiel", "umm al-fahm", "taybeh", "tira", "baqa al-gharbiyye", "shefa-amr",
          "sakhnin", "arraba", "kafr qasim", "rahat", "eilat", "yokneam", "migdal haemek", "kiryat shmona",
          "givatayim", "bnei brak", "lod", "ramla", "or yehuda", "yavne", "dimona", "safed", "tamra", "maghar",
          "london", "new york", "berlin", "dubai"]


def _term_re(term):
    t = re.escape(term.replace("-", " ")).replace(r"\ ", r"[\s-]*")       # "power bi", "Power-BI", "PowerBI"
    flags = 0 if term in AMBIGUOUS else re.I
    school = r"(?!\s+(?:university|college|institute|academic|school))" if term in CITIES else ""   # not a home town
    return re.compile(rf"(?<![\w+#.]){t.capitalize() if term in AMBIGUOUS else t}(?![\w+#]){school}", flags)


_SKILL_RES = {s.replace("-", " "): _term_re(s) for s in SKILLS}
_CITY_RES = {c: _term_re(c) for c in CITIES}


def _found(regexes, text):
    return {name for name, rx in regexes.items() if rx.search(text)}


def extract(text):
    """7a · The skills and the city found in one CV. The city is read only from the header (first 250
    characters), so "Tel Aviv University" further down is not a home town."""
    skills = _found(_SKILL_RES, text)
    header = text[:250]
    cities = [(m.start(), c) for c, rx in _CITY_RES.items() if (m := rx.search(header))]
    return skills, (min(cities)[1] if cities else None)


_NOT_A_NAME = re.compile(r"resume|curriculum|\bcv\b|profile|summary|contact|\[", re.I)


def _candidate_name(source, text):
    """The name printed at the top of the CV; the file name when no such line exists."""
    for line in text.split("\n")[:6]:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and len(line) <= 40 and not _NOT_A_NAME.search(line) \
                and not re.search(r"[\d@:/|,]", line):
            return line.title() if line.isupper() else line
    stem = re.sub(r"(?i)(cv|resume)(\W|\d|$)", " ", pathlib.Path(source).stem)    # also a glued "HuraniCV"
    return " ".join(stem.replace("_", " ").replace("-", " ").split()).title()


def build_graph(chunks):
    """7b · The knowledge graph: ("candidate", name) —has_skill— ("skill", x), —lives_in— ("city", y).
    Candidate nodes carry source=<file name>. Rebuilt on every Reload."""
    global GRAPH
    by_source = {}
    for c in chunks:                               # the "CV: <name>" prefix is not CV content: keep it out of the extraction
        by_source.setdefault(c["source"], []).append(re.sub(r"\ACV: [^\n]*\n", "", c["text"]))
    g = nx.Graph()
    for source, texts in by_source.items():
        skills, city = extract("\n".join(texts))
        name = _candidate_name(source, texts[0])
        if ("candidate", name) in g:                         # two CVs, same name: keep them apart
            name = f"{name} ({pathlib.Path(source).stem})"
        who = ("candidate", name)
        g.add_node(who, source=source)
        for s in sorted(skills):
            g.add_edge(who, ("skill", s), rel="has_skill")
        if city:
            g.add_edge(who, ("city", city), rel="lives_in")
    GRAPH = g
    return g


# 7c · answer "which candidates…" questions from the graph
_PLURAL_CUE = re.compile(r"\b(candidates|people|applicants|developers|engineers|everyone|anyone|all|list|how many)\b", re.I)


def _evidence(source, regexes, width=150):
    """The line of this CV that mentions the first entity, so the answer can show why it matched."""
    for text in COLLECTION.get(where={"source": source}, include=["documents"])["documents"]:
        for line in text.split("\n"):
            if any(rx.search(line) for rx in regexes):
                return " ".join(line.split())[:width]
    return ""


def _graph_search(query):
    """Return None when the graph does not apply, else every candidate connected to the entities in the
    question (the intersection, or the union when the question says "or"), best chunk first."""
    if GRAPH.number_of_nodes() == 0 or not _PLURAL_CUE.search(query):
        return None
    entities = [("skill", s) for s in _found(_SKILL_RES, query)] + [("city", c) for c in _found(_CITY_RES, query)]
    if not entities:
        return None
    sets = [{n for n in GRAPH.neighbors(e)} if e in GRAPH else set() for e in entities]
    people = set.union(*sets) if re.search(r"\bor\b", query, re.I) else set.intersection(*sets)
    label = " & ".join(e[1] for e in entities)
    if not people:
        return []                                            # a known skill or city that nobody has
    qvec = embed([query]).tolist()
    regexes = [(_SKILL_RES if kind == "skill" else _CITY_RES)[name] for kind, name in entities]
    hits = []
    for who in people:
        src = GRAPH.nodes[who]["source"]
        r = COLLECTION.query(query_embeddings=qvec, n_results=1, where={"source": src},
                             include=["documents", "distances"])
        hits.append({"source": src, "text": r["documents"][0][0], "score": 1 - r["distances"][0][0],
                     "vector_score": 1 - r["distances"][0][0], "via": f"graph: {label} → {who[1]}",
                     "name": who[1], "evidence": _evidence(src, regexes)})
    if RERANK:                                               # most relevant chunk first
        for h, s in zip(hits, _reranker().score(query, [h["text"] for h in hits])):
            h["score"] = s
    hits = sorted(hits, key=lambda h: -h["score"])[:MAX_GRAPH_HITS]
    return [{**h, "rank": i} for i, h in enumerate(hits, 1)]
