# Document-Based Q&A using RAG

A local Retrieval-Augmented Generation (RAG) app for asking questions over PDF documents.

## What It Does

- Ingests PDFs with page-level metadata
- Chunks text semantically
- Stores embeddings in local ChromaDB
- Uses hybrid retrieval (vector + BM25) with cross-encoder reranking
- Generates answers with citations through a Hugging Face hosted LLM
- Provides a Streamlit UI for ingestion and Q&A

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain ecosystem (`langchain`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`, `langchain-experimental`)
- ChromaDB (local persistent vector store)
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker
- Hugging Face Inference Endpoint for generation

## Project Layout

```text
.
|- app.py
|- health_check.py
|- src/
|  |- config.py
|  |- ingestion.py
|  |- retrieval.py
|  `- generator.py
|- tests/
|  |- test_ingestion.py
|  |- test_retrieval.py
|  `- test_generator.py
|- data/
|  `- chroma_db/
|- requirements.txt
`- RUN_INSTRUCTIONS.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` and set values:

```env
HUGGINGFACE_API_KEY=your_hf_token
HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-70B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
TOP_K=10
RERANK_TOP_K=3
```

## Run

Pre-flight check:

```bash
python health_check.py
```

If you want to skip remote model invocation:

```bash
python health_check.py --skip-llm
```

```bash
streamlit run app.py
```

Then:

1. Open **Document Knowledge Base** tab and ingest a PDF.
2. Open **Chat & Query** tab, start the engine, and ask questions.

## Tests

Run the minimal contract test suite:

```bash
pytest -q
```

## Notes

- Re-ingesting the same file replaces previous chunks for that file to avoid duplication.
- If no relevant context is found, the app returns a safe fallback answer.
- After ingestion, the runtime retriever/generator cache is reset so new documents are immediately available.

## Troubleshooting

- `HUGGINGFACE_API_KEY not set`: add a valid token in `.env`.
- Empty answers after ingestion: confirm the PDF has extractable text (not scanned images only).
- Slow responses: use a smaller Hugging Face model and lower `TOP_K`.
