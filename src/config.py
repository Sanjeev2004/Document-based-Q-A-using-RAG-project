import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Configuration (Required)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
# PINECONE_ENVIRONMENT is often not needed for newer indexes/client but kept for compatibility
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-document-qa")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800")) # Increased for semantic chunking base
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "10")) # Fetch more for reranking
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# System Settings
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Validate configuration
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file.")

if not HUGGINGFACE_API_KEY:
    print("Warning: HUGGINGFACE_API_KEY not set. Generation will fail.")
