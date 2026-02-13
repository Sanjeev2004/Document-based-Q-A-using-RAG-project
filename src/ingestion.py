
import os
import time
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec, PodSpec

# Import from local source config
try:
    from src.config import (
        EMBEDDING_MODEL,
        PINECONE_API_KEY,
        PINECONE_ENVIRONMENT,
        PINECONE_INDEX_NAME
    )
except ImportError:
    # Fallback for running script directly
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        EMBEDDING_MODEL,
        PINECONE_API_KEY,
        PINECONE_ENVIRONMENT,
        PINECONE_INDEX_NAME
    )

import pdfplumber

def load_pdf_with_metadata(file_path: str) -> List[Document]:
    """
    Load PDF with page numbers and filename metadata.
    """
    documents = []
    filename = os.path.basename(file_path)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": i + 1,
                            "file_path": file_path
                        }
                    ))
    except Exception as e:
        print(f"Error loading PDF with pdfplumber: {e}")
        # Fallback would go here if needed
        raise e
        
    return documents

def split_documents_semantically(documents: List[Document], embeddings) -> List[Document]:
    """
    Split documents using Semantic Chunking.
    """
    print("Splitting documents using Semantic Chunking...")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile" # or "gradient"
    )
    
    # Note: SemanticChunker.split_documents might lose some metadata in older versions,
    # but let's try standard split_documents
    chunks = text_splitter.split_documents(documents)
    return chunks

def ingest_document(file_path: str):
    """
    Ingest a document using advanced RAG techniques.
    """
    print(f"--- Starting Ingestion for {file_path} ---")
    
    # 1. Initialize Embeddings
    print(f"Initializing Embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 2. Load Document with Metadata
    print("Loading document...")
    raw_docs = load_pdf_with_metadata(file_path)
    print(f"Loaded {len(raw_docs)} pages.")
    
    # 3. Semantic Chunking
    chunks = split_documents_semantically(raw_docs, embeddings)
    print(f"Created {len(chunks)} semantic chunks.")
    
    # 4. Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index: {PINECONE_INDEX_NAME}")
        try:
             # Default heuristic for serverless
             if "gcp" in PINECONE_ENVIRONMENT or "aws" in PINECONE_ENVIRONMENT:
                 cloud = "aws" if "aws" in PINECONE_ENVIRONMENT else "gcp"
                 pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384, # standard for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=PINECONE_ENVIRONMENT)
                )
        except Exception as e:
            print(f"Fallback to PodSpec due to: {e}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=PodSpec(environment=PINECONE_ENVIRONMENT)
            )
        
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
            
    # 5. Upsert to Vector Store
    print("Upserting to Pinecone...")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("Ingestion Complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py <path_to_pdf>")
    else:
        ingest_document(sys.argv[1])
