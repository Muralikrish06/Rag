# Advanced RAG — Learning Project

Space-exploration Q&A over Wikipedia, used to learn RAG types one at a time,
each measured against the same evaluation set.

**Stack:** LangChain · Sentence-Transformers (`BAAI/bge-small-en-v1.5`) · ChromaDB · Ollama
(`qwen3.5:4b` for answers and judging, `qwen2.5:1.5b` for helper calls) · FastAPI · Streamlit

## Layout
```
naive_rag.py          Step 1: the whole pipeline in one file (load → chunk → embed → store → retrieve → generate → evaluate)
<type>_rag.py         one self-contained file per later RAG type
main.py               FastAPI backend; serves every *_rag.py (it appears in the UI dropdown automatically)
frontend/app.py       Streamlit chat UI (talks to the API over HTTP)
eval/questions.json   shared evaluation set (labels + reference answers)
notes/                concepts, production challenges, interview scenarios per step
data/                 cached Wikipedia articles + Chroma index (git-ignored)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull qwen3.5:4b && ollama pull qwen2.5:1.5b
python naive_rag.py ingest
```

## Command line
```bash
python naive_rag.py ask "Who was the commander of Apollo 13?"
python naive_rag.py eval --retrieval-only    # hit@k, recall@k, precision@k, MRR, nDCG  (~10 s)
python naive_rag.py eval                     # + faithfulness, answer relevance, correctness, refusal  (~12 min on CPU)
```

## App (API + UI)
```bash
uvicorn main:app --port 8000        # terminal 1 — Swagger at http://localhost:8000/docs
streamlit run frontend/app.py       # terminal 2 — UI at http://localhost:8501
```

## Roadmap
| Step | RAG type | Fixes | Notes |
|---|---|---|---|
| 1 | Naive RAG (baseline) | — | [notes/01_naive_rag.md](notes/01_naive_rag.md) |
| 2 | Hybrid search + Re-ranking | keyword misses, ranking | |
| 3 | Query transformation (Multi-query, HyDE, Decomposition) | vague / comparison questions | |
| 4 | Parent-child (small-to-big) | precision vs context trade-off | |
| 5 | Corrective RAG (CRAG) | irrelevant retrieval | |
| 6 | Self-RAG | hallucination, unnecessary retrieval | |
| 7 | Agentic / Multi-hop RAG (LangGraph) | multi-step reasoning | |
| 8 | Graph RAG | entity relationships | |
