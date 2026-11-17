"""Naive RAG — the whole pipeline in ONE file, top to bottom, for learning.

    STEP 1  LOAD       Wikipedia articles  -> LangChain Documents (cached on disk)
    STEP 2  CHUNK      section split + RecursiveCharacterTextSplitter
    STEP 3  EMBED      HuggingFaceEmbeddings (sentence-transformers, bge-small)
    STEP 4  STORE      Chroma (persistent, cosine)
    STEP 5  RETRIEVE   top-k similarity search + distance threshold
    STEP 6  GENERATE   prompt | ChatOllama | parser   (LCEL chain, citations, refusal)
    STEP 7  EVALUATE   retrieval metrics  (hit@k, recall@k, precision@k, MRR, nDCG@k)
                       generation metrics (faithfulness, answer relevance, correctness, refusal)

Usage:
    python naive_rag.py ingest                     # build the index (add --refresh to re-download)
    python naive_rag.py ask "Who commanded Apollo 13?"
    python naive_rag.py eval --retrieval-only      # fast: no LLM calls
    python naive_rag.py eval                       # full: retrieval + generation metrics (slow on CPU)
    python naive_rag.py eval --limit 3             # first 3 questions only

The modular version of the same pipeline (used by the FastAPI/Streamlit app) lives in
common/ + rag/naive.py. This file is intentionally self-contained: no project imports.
"""
import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "chromadb", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("naive_rag")

# =============================================================================
# CONFIG — every knob in one place
# =============================================================================
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"           # cached Wikipedia articles (shared with the modular version)
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "naive_rag_single_file_v1"   # own collection: this file never touches the app's index
EVAL_FILE = ROOT / "eval" / "questions.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"    # 384 dims, 512-token limit
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: queries only

CHUNK_SIZE = 1200        # characters (~300 tokens)
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 80

TOP_K = 5
MAX_DISTANCE = 0.45      # cosine distance; above this a chunk counts as irrelevant

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:4b"        # generates the answers
HELPER_MODEL = "qwen2.5:1.5b"   # cheap helper (answer-relevance question generation)
JUDGE_MODEL = "qwen3.5:4b"      # LLM-as-judge for faithfulness / correctness
NUM_CTX = 4096                  # Ollama default 2048 silently truncates RAG prompts

NO_ANSWER = "I don't know based on the available documents."

WIKI_TOPICS = [
    "Apollo program", "Apollo 11", "Apollo 13", "Gemini program", "Mercury program",
    "Space Shuttle program", "Space Shuttle Challenger disaster", "Space Shuttle Columbia disaster",
    "Artemis program", "International Space Station", "Mir", "Skylab",
    "Voyager 1", "Voyager 2", "Hubble Space Telescope", "James Webb Space Telescope",
    "Cassini–Huygens", "New Horizons", "Chandrayaan-3", "Mars Orbiter Mission",
    "Sojourner (rover)", "Spirit (rover)", "Opportunity (rover)", "Curiosity (rover)", "Perseverance (rover)",
    "Neil Armstrong", "Buzz Aldrin", "Yuri Gagarin", "Valentina Tereshkova", "Sally Ride", "Kalpana Chawla",
    "NASA", "Roscosmos", "Indian Space Research Organisation", "European Space Agency", "SpaceX",
    "Falcon 9", "Starship (spacecraft)", "Sputnik 1", "Space Race",
]


# =============================================================================
# STEP 1: LOAD — Wikipedia -> Documents
# =============================================================================
def fetch_article(title: str, retries: int = 3) -> dict | None:
    """Plain-text article with section headings, via the MediaWiki API."""
    params = {"action": "query", "format": "json", "formatversion": 2,
              "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
              "rvprop": "ids|timestamp", "inprop": "url", "redirects": 1, "titles": title}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30,
                                headers={"User-Agent": "rag-learning-project/0.1 (educational use)"})
            resp.raise_for_status()
            page = resp.json()["query"]["pages"][0]
            if page.get("missing"):
                return None
            return {"title": page["title"], "url": page["fullurl"],
                    "revision_id": page["revisions"][0]["revid"], "text": page["extract"]}
        except (requests.RequestException, KeyError) as e:
            log.warning("Fetch %s failed (attempt %d): %s", title, attempt, e)
            time.sleep(2 ** attempt)
    return None


def load_documents(refresh: bool = False) -> list[Document]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for title in WIKI_TOPICS:
        path = RAW_DIR / f"{re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_')}.json"
        if path.exists() and not refresh:
            raw = json.loads(path.read_text())          # cache hit: never re-download needlessly
        else:
            raw = fetch_article(title)
            if not raw:
                log.warning("Skipping %s", title)
                continue
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
            time.sleep(0.2)
        text = raw.pop("text")
        docs.append(Document(page_content=text, metadata=raw))
    return docs


# =============================================================================
# STEP 2: CHUNK — structure first, then size
# =============================================================================
SKIP_SECTIONS = {"references", "external links", "see also", "further reading", "notes",
                 "bibliography", "sources", "citations", "footnotes", "gallery"}
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],  # paragraph > line > sentence > word
    keep_separator="end",
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """'== History ==' / '=== Early years ===' -> [('History / Early years', body), ...]"""
    sections, stack, last_end, current = [], [], 0, "Introduction"
    for m in HEADING_RE.finditer(text):
        sections.append((current, text[last_end:m.start()]))
        level, name = len(m.group(1)), m.group(2)
        stack = [s for s in stack if s[0] < level] + [(level, name)]
        current = " / ".join(s[1] for s in stack)
        last_end = m.end()
    sections.append((current, text[last_end:]))
    return [(name, body.strip()) for name, body in sections
            if body.strip() and name.split(" / ")[0].lower() not in SKIP_SECTIONS]


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        title = doc.metadata["title"]
        for section, body in split_sections(doc.page_content):
            for i, text in enumerate(splitter.split_text(body)):
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(Document(
                    # Deterministic id -> re-ingesting upserts instead of duplicating.
                    id=hashlib.sha1(f"{title}|{section}|{i}".encode()).hexdigest()[:16],
                    # "Title > Section" header: the chunk stays meaningful on its own.
                    page_content=f"{title} > {section}\n{text}",
                    metadata={"title": title, "section": section, "chunk_index": i,
                              "url": doc.metadata["url"], "revision_id": doc.metadata["revision_id"]},
                ))
    return chunks


# =============================================================================
# STEP 3: EMBED — one model instance, asymmetric query/passage encoding
# =============================================================================
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},                # passages
            query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},   # queries
        )
    return _embeddings


# =============================================================================
# STEP 4: STORE — Chroma
# =============================================================================
_store = None


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(),
                        persist_directory=str(CHROMA_DIR),
                        collection_metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL})
    return _store


def ingest(refresh: bool = False) -> dict:
    t0 = time.perf_counter()
    docs = load_documents(refresh)
    chunks = chunk_documents(docs)
    log.info("Loaded %d articles -> %d chunks", len(docs), len(chunks))

    store = get_store()
    new_ids = {c.id for c in chunks}
    stale = [i for i in store.get(include=[])["ids"] if i not in new_ids]
    if stale:
        store.delete(ids=stale)            # remove chunks that no longer exist in the source
    for start in range(0, len(chunks), 256):
        batch = chunks[start:start + 256]
        store.add_documents(batch, ids=[c.id for c in batch])   # embeds + upserts
        log.info("Indexed %d/%d", min(start + 256, len(chunks)), len(chunks))
    stats = {"collection": COLLECTION, "articles": len(docs), "chunks": len(store.get(include=[])["ids"]),
             "stale_deleted": len(stale), "seconds": round(time.perf_counter() - t0, 1)}
    log.info("Done: %s", stats)
    return stats


# =============================================================================
# STEP 5: RETRIEVE
# =============================================================================
def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    results = get_store().similarity_search_with_score(question, k=k)   # (doc, cosine distance)
    docs = []
    for doc, distance in results:
        if distance <= MAX_DISTANCE:         # guardrail: drop weak matches
            doc.metadata["distance"] = round(distance, 4)
            docs.append(doc)
    return docs


# =============================================================================
# STEP 6: GENERATE
# =============================================================================
def get_llm(model: str = LLM_MODEL, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=temperature, num_ctx=NUM_CTX,
                      reasoning=False, client_kwargs={"timeout": 180})


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a space-exploration assistant. Answer the question using ONLY the numbered context passages.
Rules:
- Cite passages inline like [1] or [2][3] for every fact you state.
- If the context does not contain the answer, reply exactly: "{no_answer}"
- Do not use outside knowledge. Be concise."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]).partial(no_answer=NO_ANSWER)

_rag_chain = None


def answer(question: str, k: int = TOP_K) -> dict:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAG_PROMPT | get_llm().with_retry(stop_after_attempt=3) | StrOutputParser()

    t0 = time.perf_counter()
    docs = retrieve(question, k)
    timings = {"retrieve_s": round(time.perf_counter() - t0, 3)}
    if not docs:                              # nothing relevant -> refuse without calling the LLM
        return {"question": question, "answer": NO_ANSWER, "docs": [], "timings": timings}

    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
    t0 = time.perf_counter()
    text = _rag_chain.invoke({"context": context, "question": question})
    timings["generate_s"] = round(time.perf_counter() - t0, 3)
    return {"question": question, "answer": text, "docs": docs, "timings": timings}


# =============================================================================
# STEP 7a: RETRIEVAL METRICS — need only labels (which articles are relevant), no LLM
# =============================================================================
# Relevance is judged at article level: a chunk is relevant if it comes from an expected article.
def retrieval_metrics(retrieved_titles: list[str], expected: set[str], k: int = TOP_K) -> dict:
    """
    hit@k       : 1 if ANY relevant chunk is in the top-k                 ("did we find something?")
    recall@k    : fraction of expected articles present in the top-k      ("did we find everything?")
    precision@k : fraction of the k slots holding relevant chunks         ("how much noise?")
    MRR         : 1 / rank of the first relevant chunk                    ("how high is the first hit?")
    nDCG@k      : rank-discounted gain, each expected article counted once, normalised by the ideal ranking
    """
    if not expected:   # out-of-scope question: correct behaviour is to retrieve nothing
        ok = float(not retrieved_titles)
        return {"hit@k": ok, "recall@k": ok, "precision@k": ok, "mrr": ok, "ndcg@k": ok}

    relevant = [t in expected for t in retrieved_titles]
    first = next((i for i, r in enumerate(relevant, 1) if r), None)

    seen, dcg = set(), 0.0
    for rank, title in enumerate(retrieved_titles, 1):
        if title in expected and title not in seen:
            seen.add(title)
            dcg += 1 / math.log2(rank + 1)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(k, len(expected)) + 1))

    return {
        "hit@k": float(first is not None),
        "recall@k": len(seen) / len(expected),
        "precision@k": sum(relevant) / k,
        "mrr": 1 / first if first else 0.0,
        "ndcg@k": dcg / idcg,
    }


# =============================================================================
# STEP 7b: GENERATION METRICS — LLM-as-judge (RAGAS-style), plus deterministic checks
# =============================================================================
class FaithfulnessVerdict(BaseModel):
    claims: list[str] = Field(description="atomic factual claims made in the answer")
    supported: list[bool] = Field(description="for each claim, true if the context supports it")


class CorrectnessVerdict(BaseModel):
    score: float = Field(description="1.0 fully correct, 0.5 partially correct, 0.0 wrong or missing")
    reason: str


class GeneratedQuestions(BaseModel):
    questions: list[str]


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact-checker. Break the ANSWER into short atomic factual claims "
               "(ignore citation markers like [1]). For each claim decide if the CONTEXT explicitly supports it. "
               "Outside knowledge does not count as support."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])
CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You grade answers against a reference answer. Judge only factual agreement on what the "
               "question asks; ignore style, length and citation markers."),
    ("human", "QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}"),
])
REVERSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Write 3 different questions that the given answer would be a direct answer to."),
    ("human", "ANSWER: {answer}"),
])


def is_refusal(text: str) -> bool:
    return "don't know" in text.lower() or "do not know" in text.lower()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # embeddings are normalised -> dot product == cosine


def generation_metrics(q: dict, result: dict) -> dict:
    """
    faithfulness     : supported claims / total claims, judged against the RETRIEVED context
                       (catches hallucination — can be 1.0 even if the answer is wrong, if context was wrong)
    answer_relevance : RAGAS approach — generate questions from the answer, embed them, mean cosine
                       to the original question ("does the answer address what was asked?")
    correctness      : judge compares answer with the human reference answer
    keyword_match    : deterministic sanity check — expected keywords appear in the answer
    refusal_ok       : out-of-scope -> must refuse; in-scope -> must not refuse
    """
    ans = result["answer"]
    out_of_scope = not q["expected_titles"]
    refused = is_refusal(ans)
    m = {
        "keyword_match": float(all(s.lower() in ans.lower() for s in q["answer_contains"])),
        "refusal_ok": float(refused == out_of_scope),
    }
    if refused:
        # A refusal makes no claims (faithful by definition) and isn't scored for relevance/correctness:
        # refusal_ok already says whether refusing was the right call.
        m.update({"faithfulness": 1.0, "answer_relevance": None,
                  "correctness": 1.0 if out_of_scope else 0.0})
        return m

    judge = get_llm(JUDGE_MODEL, temperature=0.0)
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(result["docs"], 1))

    v = (FAITHFULNESS_PROMPT | judge.with_structured_output(FaithfulnessVerdict)).invoke(
        {"context": context, "answer": ans})
    m["faithfulness"] = sum(v.supported) / len(v.supported) if v.supported else 0.0
    m["unsupported_claims"] = [c for c, s in zip(v.claims, v.supported) if not s]

    c = (CORRECTNESS_PROMPT | judge.with_structured_output(CorrectnessVerdict)).invoke(
        {"question": q["question"], "reference": q["reference"], "answer": ans})
    m["correctness"] = max(0.0, min(1.0, c.score))

    gq = (REVERSE_QUESTION_PROMPT | get_llm(HELPER_MODEL, 0.0).with_structured_output(GeneratedQuestions)).invoke(
        {"answer": ans})
    if gq.questions:
        emb = get_embeddings()
        qv = emb.embed_query(q["question"])
        m["answer_relevance"] = sum(cosine(qv, emb.embed_query(g)) for g in gq.questions) / len(gq.questions)
    else:
        m["answer_relevance"] = 0.0
    return m


# =============================================================================
# STEP 7c: EVALUATION RUNNER
# =============================================================================
def mean(values: list) -> str:
    vals = [v for v in values if v is not None]
    return f"{sum(vals) / len(vals):.2f}" if vals else "  - "


def evaluate(retrieval_only: bool = False, limit: int | None = None) -> None:
    questions = json.loads(EVAL_FILE.read_text())[:limit]
    rows = []
    for n, q in enumerate(questions, 1):
        docs = retrieve(q["question"])
        titles = [d.metadata["title"] for d in docs]
        row = {"id": q["id"], "type": q["type"], **retrieval_metrics(titles, set(q["expected_titles"]))}
        line = f"[{n}/{len(questions)}] {q['id']:<11} hit={row['hit@k']:.0f} recall={row['recall@k']:.2f}"

        if not retrieval_only:
            t0 = time.perf_counter()
            result = answer(q["question"])
            row["latency_s"] = sum(result["timings"].values())
            row.update(generation_metrics(q, result))
            line += (f" | faith={row['faithfulness']:.2f} correct={row['correctness']:.2f} "
                     f"refusal_ok={row['refusal_ok']:.0f} ({time.perf_counter() - t0:.0f}s)")
            if row.get("unsupported_claims"):
                line += f"\n      unsupported: {row['unsupported_claims']}"
        print(line, flush=True)
        rows.append(row)

    ret_cols = ["hit@k", "recall@k", "precision@k", "mrr", "ndcg@k"]
    gen_cols = [] if retrieval_only else ["faithfulness", "answer_relevance", "correctness", "keyword_match",
                                          "refusal_ok", "latency_s"]
    short = {"precision@k": "prec@k", "faithfulness": "faithful", "answer_relevance": "ans_rel",
             "correctness": "correct", "keyword_match": "keyword", "refusal_ok": "refusal", "latency_s": "latency"}
    header = f"\n{'type':<13}{'n':>3}" + "".join(f"{short.get(c, c):>10}" for c in ret_cols + gen_cols)
    print(header + "\n" + "-" * (len(header) - 1))
    groups = {}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)
    for name, group in [*groups.items(), ("ALL", rows)]:
        print(f"{name:<13}{len(group):>3}" + "".join(f"{mean([r.get(c) for r in group]):>10}"
                                                     for c in ret_cols + gen_cols))
    if not retrieval_only:
        print(f"\nJudge: {JUDGE_MODEL} (same family as the generator -> expect some self-preference bias)")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("ask")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=TOP_K)
    p = sub.add_parser("eval")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.refresh)
    elif args.cmd == "ask":
        if not get_store().get(limit=1)["ids"]:
            sys.exit("Index is empty: run `python naive_rag.py ingest` first.")
        r = answer(" ".join(args.question), args.k)
        print(f"\nQ: {r['question']}\n\nA: {r['answer']}\n\nSources:")
        for i, d in enumerate(r["docs"], 1):
            print(f"  [{i}] {d.metadata['title']} > {d.metadata['section']}  (distance={d.metadata['distance']})")
        print(f"\nTimings: {r['timings']}")
    elif args.cmd == "eval":
        evaluate(args.retrieval_only, args.limit)


if __name__ == "__main__":
    main()
