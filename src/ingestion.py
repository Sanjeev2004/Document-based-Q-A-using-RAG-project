import os
from typing import Dict, List, Optional

import pdfplumber
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from src.vectorstore import get_chroma_vectorstore

try:
    from src.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY


def load_pdf_with_metadata(file_path: str, source_name: Optional[str] = None) -> List[Document]:
    """
    Load PDF with page numbers and filename metadata.
    """
    documents = []
    filename = source_name or os.path.basename(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": i + 1,
                                "file_path": file_path,
                            },
                        )
                    )
    except Exception as exc:
        print(f"Error loading PDF with pdfplumber: {exc}")
        raise

    return documents


def split_documents_semantically(documents: List[Document], embeddings) -> List[Document]:
    """
    Split documents using Semantic Chunking.
    """
    print("Splitting documents using Semantic Chunking...")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )
    return text_splitter.split_documents(documents)


def _ingest_document_with_resources(
    file_path: str,
    embeddings,
    vectorstore,
    source_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Ingest a single PDF using already-initialized embeddings/vectorstore.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported for ingestion.")

    print(f"--- Starting Ingestion for {file_path} ---")
    filename = source_name or os.path.basename(file_path)

    print("Loading document...")
    raw_docs = load_pdf_with_metadata(file_path, source_name=filename)
    if not raw_docs:
        raise ValueError("No extractable text found in the PDF.")
    print(f"Loaded {len(raw_docs)} pages.")

    chunks = split_documents_semantically(raw_docs, embeddings)
    print(f"Created {len(chunks)} semantic chunks.")

    print(f"Upserting to ChromaDB at {CHROMA_PERSIST_DIRECTORY}...")
    vectorstore.delete(where={"source": filename})

    ids = []
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        ids.append(f"{filename}:{chunk.metadata.get('page', 'na')}:{idx}")

    vectorstore.add_documents(chunks, ids=ids)
    print("Ingestion Complete!")
    return {"pages": len(raw_docs), "chunks": len(chunks)}


def ingest_document(file_path: str, source_name: Optional[str] = None) -> Dict[str, int]:
    """
    Ingest a document using advanced RAG techniques (ChromaDB).
    """
    print(f"Initializing Embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Connecting ChromaDB at {CHROMA_PERSIST_DIRECTORY}...")
    vectorstore = get_chroma_vectorstore(embeddings, allow_repair=True)
    return _ingest_document_with_resources(
        file_path=file_path,
        embeddings=embeddings,
        vectorstore=vectorstore,
        source_name=source_name,
    )


def ingest_documents(file_paths: List[str], source_names: Optional[List[str]] = None) -> Dict[str, object]:
    """
    Batch ingest multiple PDFs with shared embeddings/vectorstore initialization.
    """
    if not file_paths:
        return {"ingested": [], "failed": [], "total_chunks": 0}

    if source_names and len(source_names) != len(file_paths):
        raise ValueError("source_names length must match file_paths length.")

    print(f"Initializing Embeddings once for batch: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Connecting ChromaDB once for batch at {CHROMA_PERSIST_DIRECTORY}...")
    vectorstore = get_chroma_vectorstore(embeddings, allow_repair=True)

    ingested = []
    failed = []
    total_chunks = 0

    for idx, file_path in enumerate(file_paths):
        source_name = source_names[idx] if source_names else None
        try:
            stats = _ingest_document_with_resources(
                file_path=file_path,
                embeddings=embeddings,
                vectorstore=vectorstore,
                source_name=source_name,
            )
            total_chunks += stats["chunks"]
            ingested.append(
                {
                    "file_path": file_path,
                    "source": source_name or os.path.basename(file_path),
                    "pages": stats["pages"],
                    "chunks": stats["chunks"],
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "file_path": file_path,
                    "source": source_name or os.path.basename(file_path),
                    "error": str(exc),
                }
            )

    return {"ingested": ingested, "failed": failed, "total_chunks": total_chunks}


def clear_vectorstore() -> int:
    """
    Remove all documents from the current ChromaDB collection.
    Returns number of deleted records.
    """
    print(f"Initializing Embeddings for clear operation: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = get_chroma_vectorstore(embeddings, allow_repair=True)
    data = vectorstore.get()
    ids = data.get("ids", []) if data else []
    if ids:
        vectorstore.delete(ids=ids)
    print(f"Cleared vectorstore documents: {len(ids)}")
    return len(ids)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py <path_to_pdf>")
    else:
        ingest_document(sys.argv[1])
