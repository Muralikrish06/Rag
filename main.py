"""FastAPI backend serving every single-file RAG implementation (*_rag.py) in this folder.

    POST /ingest          build the index for a RAG type        (runs in background)
    GET  /ingest/status   progress / result of the last ingestion
    POST /ask             question -> answer + sources + timings
    GET  /rag-types       available RAG types (naive_rag.py -> "naive", hybrid_rag.py -> "hybrid", ...)
    GET  /health          Ollama + vector store status

Run:   uvicorn main:app --port 8000          (docs at http://localhost:8000/docs)
UI:    streamlit run frontend/app.py

Contract each <name>_rag.py must follow:
    ingest(refresh: bool) -> dict
    answer(question: str, k: int) -> {"question", "answer", "docs": [Document], "timings": {...}}
    get_store() -> vector store;   LLM_MODEL, HELPER_MODEL, EMBED_MODEL, OLLAMA_URL, TOP_K constants

Production notes:
- Heavy objects (embedding model, Chroma client) load once at startup, not per request.
- Ingestion takes minutes: it runs as a background task, guarded by a lock so two ingestions
  never run concurrently. Clients poll /ingest/status.
- Endpoints are plain `def`: FastAPI runs them in a threadpool, so a slow LLM call doesn't
  block the event loop for other requests.
- Downstream failures (Ollama down/slow) map to 503, bad input to 4xx - never a bare 500.
"""
import importlib
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("api")

ROOT = Path(__file__).parent
# Allowlist of RAG modules discovered on disk: the API never imports an arbitrary name from a request.
RAG_TYPES = sorted(p.stem.removesuffix("_rag") for p in ROOT.glob("*_rag.py"))
DEFAULT_RAG = "naive"


def _load_rag(name: str):
    if name not in RAG_TYPES:
        raise HTTPException(404, f"Unknown rag_type '{name}'. Available: {RAG_TYPES}")
    return importlib.import_module(f"{name}_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    for name in RAG_TYPES:
        _load_rag(name).get_store().similarity_search("warm up", k=1)  # loads embedding model + opens Chroma
    log.info("Startup complete in %.1fs; RAG types: %s", time.perf_counter() - t0, RAG_TYPES)
    yield


app = FastAPI(title="Space RAG API", version="0.2.0", lifespan=lifespan)


# --- Schemas ----------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    rag_type: str = DEFAULT_RAG
    top_k: int = Field(5, ge=1, le=20)


class Source(BaseModel):
    n: int
    title: str
    section: str
    url: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    rag_type: str
    sources: list[Source]
    timings: dict[str, float]


class IngestRequest(BaseModel):
    rag_type: str = DEFAULT_RAG
    refresh: bool = Field(False, description="re-download articles from Wikipedia")


# --- Ingestion --------------------------------------------------------------
_ingest_lock = threading.Lock()
_ingest_status: dict = {"state": "idle"}


def _ingest_job(rag, refresh: bool) -> None:
    global _ingest_status
    try:
        stats = rag.ingest(refresh=refresh)
        _ingest_status = {**_ingest_status, "state": "done",
                          "finished_at": datetime.now(timezone.utc).isoformat(), **stats}
    except Exception as e:  # noqa: BLE001 - report any failure to the client
        log.exception("Ingestion failed")
        _ingest_status = {**_ingest_status, "state": "failed", "error": str(e)}
    finally:
        _ingest_lock.release()


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks):
    global _ingest_status
    rag = _load_rag(req.rag_type)
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "Ingestion already running")
    _ingest_status = {"state": "running", "rag_type": req.rag_type,
                      "started_at": datetime.now(timezone.utc).isoformat()}
    background.add_task(_ingest_job, rag, req.refresh)
    return _ingest_status


@app.get("/ingest/status")
def ingest_status():
    return _ingest_status


# --- Query ------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    rag = _load_rag(req.rag_type)
    t0 = time.perf_counter()
    try:
        result = rag.answer(req.question.strip(), k=req.top_k)
    except Exception as e:  # noqa: BLE001
        # Connection refused / timeout from Ollama -> the dependency is unavailable, not our bug.
        if "connect" in str(e).lower() or "timeout" in str(e).lower():
            log.error("LLM unavailable: %s", e)
            raise HTTPException(503, f"LLM backend unavailable: {e}") from e
        log.exception("ask failed")
        raise HTTPException(500, "Internal error while answering") from e
    timings = {**result["timings"], "total_s": round(time.perf_counter() - t0, 3)}
    sources = [{"n": i, "title": d.metadata["title"], "section": d.metadata["section"],
                "url": d.metadata["url"], "distance": d.metadata["distance"]}
               for i, d in enumerate(result["docs"], 1)]
    log.info("ask rag=%s k=%d sources=%d timings=%s q=%r", req.rag_type, req.top_k, len(sources),
             timings, req.question[:80])
    return {"question": result["question"], "answer": result["answer"], "rag_type": req.rag_type,
            "sources": sources, "timings": timings}


@app.get("/rag-types")
def rag_types():
    return {"rag_types": RAG_TYPES, "default": DEFAULT_RAG}


@app.get("/health")
def health(rag_type: str = DEFAULT_RAG):
    rag = _load_rag(rag_type)
    try:
        models = [m["name"] for m in requests.get(f"{rag.OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
        ollama = {"ok": rag.LLM_MODEL in models, "model": rag.LLM_MODEL, "helper_model": rag.HELPER_MODEL}
    except requests.RequestException as e:
        ollama = {"ok": False, "error": str(e)}
    store = rag.get_store()
    chunks = len(store.get(include=[])["ids"])
    return {"ok": ollama["ok"] and chunks > 0, "rag_type": rag_type, "ollama": ollama,
            "vector_store": {"ok": chunks > 0, "collection": store._collection.name, "chunks": chunks},
            "embed_model": rag.EMBED_MODEL}
