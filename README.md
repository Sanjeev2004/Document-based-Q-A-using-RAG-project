# Document-Based Q&A using RAG

[![CI](https://github.com/Sanjeev2004/Document-based-Q-A-using-RAG-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanjeev2004/Document-based-Q-A-using-RAG-project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)

A production-style local RAG application for answering questions from PDFs with citations and robust retrieval controls.

## Why This Project Stands Out

- Hybrid retrieval pipeline: vector search + BM25 + cross-encoder reranking.
- Source-aware querying: restrict answers to latest uploaded documents.
- Recovery-first design: Chroma corruption detection + auto-repair flow.
- Batch ingestion support with per-file success/failure reporting.
- Fast feedback loop: health checks + automated CI + tests.

## Key Outcomes

- `11` automated tests passing (`pytest -q`).
- End-to-end health verification command (`python health_check.py`).
- Stable local vector store lifecycle:
  - clear/replace knowledge base
  - refresh runtime retriever/generator cache
  - isolate and recover broken Chroma state

## Demo Flow

1. Upload one or more PDFs in **Document Knowledge Base**.
2. Choose whether to replace previous index entries.
3. Query in **Chat & Query**.
4. Optionally enforce **latest uploaded files only** to avoid stale answers.
5. Inspect reranked citation evidence (source + page + score).

## Architecture

```text
PDF(s)
  -> page-level extraction (pdfplumber)
  -> semantic chunking (LangChain Experimental)
  -> embeddings (sentence-transformers)
  -> local ChromaDB index
  -> retrieval (vector + BM25)
  -> cross-encoder rerank (MS MARCO MiniLM)
  -> answer generation (Hugging Face chat completion)
  -> cited response in Streamlit
```

## Engineering Decisions

- **ChromaDB (local)** over managed vector DB:
  - better for portfolio reproducibility and zero infra setup.
- **Hybrid retrieval + reranking** over plain similarity search:
  - improves precision on mixed keyword/semantic queries.
- **Source filtering**:
  - prevents old PDFs from dominating answers after new ingestion.
- **Health checks and tests in repo**:
  - signals engineering rigor to recruiters and contributors.

## Project Layout

```text
.
|- app.py
|- health_check.py
|- src/
|  |- config.py
|  |- vectorstore.py
|  |- ingestion.py
|  |- retrieval.py
|  `- generator.py
|- tests/
|- .env.example
|- requirements.txt
`- RUN_INSTRUCTIONS.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template:

```bash
cp .env.example .env
```

4. Add your `HUGGINGFACE_API_KEY` in `.env`.

## Run

```bash
python health_check.py
streamlit run app.py
```

If you want local-only checks:

```bash
python health_check.py --skip-llm
```

## Tests and CI

- Local:

```bash
pytest -q
```

- GitHub Actions CI (`.github/workflows/ci.yml`) runs on push/PR to `main`.

## Failure Modes and Recovery

- Chroma Rust panic / tenant errors:
  - auto-repair utility moves broken DB to timestamped backup and recreates a fresh store.
- Outdated retrieval results:
  - replace knowledge base during ingestion and/or enable latest-source query filter.
- Missing LLM access:
  - health check surfaces token/model issues before app startup.

## Resume-Ready Summary

Built a robust document Q&A RAG system using Python, LangChain, ChromaDB, and Hugging Face. Implemented hybrid retrieval with cross-encoder reranking, source-scoped querying, batch ingestion, health checks, and CI-tested reliability for production-like behavior.
