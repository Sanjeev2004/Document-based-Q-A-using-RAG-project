"""
Document ingestion script
Loads documents, chunks them, generates embeddings, and stores in Pinecone vector database
"""

import os
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.docstore.document import Document
import pinecone

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME
)
from utils import load_document, split_text


def initialize_embeddings():
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_pinecone_vectorstore(chunks: list, embeddings):
    """
    Create or update Pinecone vector store.
    
    Args:
        chunks: List of text chunks
        embeddings: Embedding model
        
    Returns:
        Pinecone vector store
    """
    # Initialize Pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    
    # Convert chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Check if index exists, create if not
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # Create index with appropriate dimensions (all-MiniLM-L6-v2 has 384 dimensions)
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine"
        )
    
    # Create or load vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    return vectorstore


def ingest_document(file_path: str) -> None:
    """
    Main ingestion function: load, chunk, embed, and store documents.
    
    Args:
        file_path: Path to the document file
    """
    print(f"Loading document: {file_path}")
    
    # Load document
    text = load_document(file_path)
    
    if not text.strip():
        raise ValueError("Document is empty or could not be extracted")
    
    print(f"Document loaded. Text length: {len(text)} characters")
    
    # Split into chunks
    print(f"Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")
    
    # Initialize embeddings
    print(f"Initializing embeddings model: {EMBEDDING_MODEL}...")
    embeddings = initialize_embeddings()
    
    # Create vector store
    print("Creating/updating Pinecone vector store...")
    vectorstore = create_pinecone_vectorstore(chunks, embeddings)
    print("Pinecone vector store updated successfully!")
    
    print("Document ingestion completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_document>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        ingest_document(file_path)
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)
