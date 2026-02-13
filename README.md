# Document-Based Question Answering using RAG

A production-quality Retrieval-Augmented Generation (RAG) system for answering questions from document collections. Built entirely with free and open-source tools, this system enables semantic search and context-aware question answering over your documents.

## Project Overview

This project implements a complete RAG pipeline that combines document retrieval with large language model generation. The system processes PDF documents, creates semantic embeddings, stores them in a vector database, and uses them to provide accurate, context-grounded answers to user questions.

### Key Features

- **Document Processing**: Automatic PDF text extraction and intelligent chunking
- **Semantic Search**: Vector-based similarity search for relevant document chunks
- **Context-Aware Answers**: LLM-generated responses grounded in retrieved documents
- **Pinecone Vector Store**: Cloud-based managed vector database for scalable document storage
- **Streamlit UI**: Intuitive web interface for document upload and Q&A
- **Production Ready**: Modular architecture with error handling and configuration management

## Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │
│ (500 chars, 50  │
│    overlap)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
│ (sentence-      │
│ transformers)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Vector Store   │◄─────┤   Pinecone   │
│                 │      │   (Cloud)    │
└────────┬────────┘      └──────────────┘
         │
         │ User Question
         ▼
┌─────────────────┐
│  Similarity     │
│  Search (Top-K) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context +      │
│  Question       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hugging Face   │
│      LLM        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

## Tech Stack

### Core Technologies

- **Python 3.8+**: Primary programming language
- **LangChain**: Framework for LLM applications and RAG pipelines
- **Streamlit**: Web UI framework
- **Hugging Face**: LLM inference API and embedding models

### LLM & Embeddings

- **LLM**: Hugging Face Inference API (Mistral-7B-Instruct or compatible models)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

### Vector Databases

- **Pinecone**: Managed cloud vector database service for scalable document storage and retrieval

### Document Processing

- **PyPDF2**: PDF text extraction
- **RecursiveCharacterTextSplitter**: Intelligent text chunking

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Hugging Face account (free tier sufficient)
- Pinecone account (free tier available)

### Step-by-Step Installation

1. **Clone or download this repository**

```bash
cd "Document-Based Q&A using RAG"
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Copy the example environment file and configure it:

```bash
copy .env.example .env
```

Edit `.env` and add your API keys:

```
HUGGINGFACE_API_KEY=your_actual_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

To get a Hugging Face API key:
- Go to https://huggingface.co/settings/tokens
- Create a new token (read access is sufficient)
- Copy the token to your `.env` file

To get Pinecone credentials:
- Sign up at https://www.pinecone.io/ (free tier available)
- Create a new project and index
- Copy your API key and environment from the Pinecone dashboard
- Add them to your `.env` file

## Environment Variables

The following environment variables can be configured in your `.env` file:

### Required

- `HUGGINGFACE_API_KEY`: Your Hugging Face API token for LLM inference
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., `us-east1-gcp`)

### Optional

- `HUGGINGFACE_MODEL`: LLM model to use (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `PINECONE_INDEX_NAME`: Name for the Pinecone index (default: `rag-document-qa`)
- `CHUNK_SIZE`: Text chunk size in characters (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)
- `TOP_K`: Number of chunks to retrieve (default: `3`)

## How to Run

### Step 1: Ingest Documents

Before asking questions, you need to ingest at least one document into the vector store.

**Option A: Using the command line**

```bash
python ingest.py data/sample.pdf
```

Replace `data/sample.pdf` with the path to your PDF file.

**Option B: Using the Streamlit UI**

See Step 2 below - the UI includes a document ingestion tab.

### Step 2: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Step 3: Use the Application

1. **Document Ingestion Tab**: Upload PDF files or provide file paths to add documents to the knowledge base
2. **Question Answering Tab**: Enter questions about your ingested documents and receive answers with source citations

## FAISS vs Pinecone

### FAISS (Default)

**Advantages:**
- Free and open-source
- Runs locally, no internet required after setup
- Fast similarity search
- No API limits or costs
- Full control over data

**Disadvantages:**
- Requires local storage
- Not suitable for distributed systems
- Manual backup required

**Best for:** Local development, single-user applications, cost-sensitive projects

### Pinecone

**Advantages:**
- Managed service, no infrastructure management
- Scalable to millions of vectors
- Built-in redundancy and backups
- API access from anywhere
- Real-time updates

**Disadvantages:**
- Requires internet connection
- Free tier has limitations
- Potential costs at scale

**Best for:** Production deployments, multi-user applications, cloud-based systems

**Switching between them:**

Edit `config.py` or set `USE_PINECONE=true` in your `.env` file to switch to Pinecone. Ensure you have configured all Pinecone-related environment variables.

## Example Questions

After ingesting documents, try asking:

- "What is the main topic of this document?"
- "Summarize the key points discussed"
- "What are the conclusions?"
- "Explain [specific concept] mentioned in the document"
- "What data or statistics are provided?"

The system will retrieve relevant chunks and generate context-aware answers.

## Project Structure

```
rag-document-qa/
│
├── app.py                 # Streamlit web application
├── ingest.py              # Document ingestion script
├── rag_pipeline.py        # Core RAG pipeline logic
├── config.py              # Configuration and environment setup
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
│
├── data/                  # Sample documents directory
│   └── sample.pdf
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── loaders.py         # Document loaders
│   ├── splitter.py        # Text chunking
│   └── helpers.py         # Helper functions
│
└── README.md              # This file
```

## Resume-Ready Project Description

**Document-Based Question Answering System using RAG**

Developed a production-quality Retrieval-Augmented Generation (RAG) system for semantic document search and question answering. The system processes PDF documents, generates embeddings using sentence transformers, and stores them in Pinecone vector database. Implemented a complete pipeline using LangChain that retrieves relevant document chunks and generates context-grounded answers using Hugging Face LLMs. Built an intuitive Streamlit web interface for document upload and interactive Q&A. The modular architecture is fully configurable via environment variables and demonstrates expertise in NLP, cloud vector databases, LLM integration, and full-stack Python development.

**Key Technologies:** Python, LangChain, Hugging Face, Pinecone, Streamlit, Sentence Transformers, RAG, Vector Databases, LLM APIs

## Future Improvements

### Short-term Enhancements

- Support for additional document formats (DOCX, TXT, Markdown)
- Batch document ingestion
- Document metadata extraction and filtering
- Answer confidence scoring
- Conversation history and follow-up questions

### Medium-term Features

- Multi-document knowledge bases with document selection
- Advanced chunking strategies (semantic chunking, hierarchical)
- Hybrid search (keyword + semantic)
- Custom embedding models fine-tuned on domain data
- Export/import vector stores

### Long-term Vision

- Multi-modal support (images, tables)
- Real-time document updates
- User authentication and document access control
- API endpoints for programmatic access
- Docker containerization
- Deployment guides for cloud platforms

## Troubleshooting

### Common Issues

**Issue: "HUGGINGFACE_API_KEY not found"**
- Solution: Ensure your `.env` file exists and contains a valid Hugging Face API key

**Issue: "PINECONE_API_KEY is required" or "PINECONE_ENVIRONMENT is required"**
- Solution: Ensure your `.env` file contains both `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` with valid values from your Pinecone dashboard

**Issue: "Pinecone index not found"**
- Solution: The index will be created automatically when you run `python ingest.py <document_path>`. Ensure your Pinecone API key has permission to create indexes.

**Issue: "Model not found" or API errors**
- Solution: Verify your Hugging Face API key has access to the model. Some models may require acceptance of terms on Hugging Face.

**Issue: Slow response times**
- Solution: Consider using a smaller/faster model or reducing TOP_K value. Pinecone free tier may have rate limits.

**Issue: Memory errors with large documents**
- Solution: Reduce CHUNK_SIZE or process documents in batches

**Issue: Pinecone connection errors**
- Solution: Verify your internet connection and check that your Pinecone API key and environment are correct. Ensure your Pinecone account is active.

## License

This project is open-source and available for educational and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## Acknowledgments

- LangChain team for the excellent framework
- Hugging Face for open-source models and infrastructure
- Pinecone for the managed vector database service
- Streamlit for the UI framework
