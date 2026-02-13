import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800")) # Increased for semantic chunking base
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "10")) # Fetch more for reranking
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# System Settings
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

if not HUGGINGFACE_API_KEY:
    print("Warning: HUGGINGFACE_API_KEY not set. Generation will fail.")
