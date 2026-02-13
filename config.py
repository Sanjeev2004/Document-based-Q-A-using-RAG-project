"""
Configuration file for RAG Document Q&A System
Uses Pinecone as the vector store
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Configuration (Required)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-document-qa")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "3"))

# Validate configuration
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file.")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT is required. Please set it in your .env file.")

if not HUGGINGFACE_API_KEY:
    print("Warning: HUGGINGFACE_API_KEY not set. Some features may not work.")
