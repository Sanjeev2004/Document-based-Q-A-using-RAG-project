"""
Streamlit UI for Document-based Q&A using RAG
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

from rag_pipeline import get_rag_pipeline, RAGPipeline
from ingest import ingest_document
from config import PINECONE_INDEX_NAME
from utils import is_valid_pdf


# Page configuration
st.set_page_config(
    page_title="Document Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False


def check_vectorstore_exists():
    """Check if vector store exists."""
    # For Pinecone, we'll try to initialize and catch errors
    return True  # Will be validated when pipeline initializes


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    try:
        with st.spinner("Initializing RAG pipeline..."):
            pipeline = get_rag_pipeline()
            st.session_state.rag_pipeline = pipeline
            st.session_state.vectorstore_ready = True
            return True
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        st.info("Please ensure you have ingested documents first using the 'Document Ingestion' tab.")
        return False


# Main UI
st.markdown('<div class="main-header">📚 Document-Based Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Retrieval-Augmented Generation (RAG)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.info(f"Vector Store: Pinecone")
    
    if st.button("Check Vector Store Status"):
        if check_vectorstore_exists():
            st.success("Vector store is ready!")
        else:
            st.warning("Vector store not found. Please ingest documents first.")
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This application uses RAG (Retrieval-Augmented Generation) to answer questions
    based on your uploaded documents.
    
    **Features:**
    - Upload PDF documents
    - Automatic text chunking
    - Semantic search
    - Context-aware answers
    """)

# Tabs
tab1, tab2 = st.tabs(["📖 Question Answering", "📄 Document Ingestion"])

# Tab 1: Question Answering
with tab1:
    st.header("Ask Questions About Your Documents")
    
    # Check if vector store exists
    if not check_vectorstore_exists():
        st.warning("⚠️ No vector store found. Please ingest documents first using the 'Document Ingestion' tab.")
    else:
        # Initialize pipeline if not already done
        if st.session_state.rag_pipeline is None:
            if st.button("Initialize RAG Pipeline"):
                initialize_pipeline()
        
        if st.session_state.rag_pipeline is not None:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?",
                key="question_input"
            )
            
            if st.button("Get Answer", type="primary") or question:
                if question:
                    try:
                        with st.spinner("Searching documents and generating answer..."):
                            result = st.session_state.rag_pipeline.answer_question(question)
                            
                            # Display answer
                            st.markdown("### Answer")
                            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                            
                            # Display source documents
                            if result.get("source_documents"):
                                st.markdown("---")
                                st.markdown("### Source Documents")
                                
                                for i, doc in enumerate(result["source_documents"], 1):
                                    with st.expander(f"Source {i}"):
                                        st.text(doc["content"])
                                        if doc.get("metadata"):
                                            st.caption(f"Metadata: {doc['metadata']}")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                else:
                    st.info("Please enter a question to get an answer.")
        else:
            st.info("Click 'Initialize RAG Pipeline' to start asking questions.")

# Tab 2: Document Ingestion
with tab2:
    st.header("Upload and Ingest Documents")
    
    st.markdown("""
    Upload a PDF document to add it to the knowledge base.
    The document will be processed, chunked, and indexed for question answering.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("Ingest Document", type="primary"):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Ingest document
                with st.spinner("Processing document... This may take a few moments."):
                    ingest_document(tmp_path)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success("✅ Document ingested successfully!")
                st.info("You can now ask questions about this document in the 'Question Answering' tab.")
                
                # Reset pipeline to reload vector store
                st.session_state.rag_pipeline = None
                st.session_state.vectorstore_ready = False
                
            except Exception as e:
                st.error(f"Error ingesting document: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Option to ingest from file path
    st.markdown("---")
    st.subheader("Or ingest from file path")
    
    file_path = st.text_input(
        "Enter file path:",
        placeholder="e.g., data/sample.pdf",
        help="Enter the path to a PDF file on your system"
    )
    
    if st.button("Ingest from Path"):
        if file_path and os.path.exists(file_path):
            if is_valid_pdf(file_path):
                try:
                    with st.spinner("Processing document... This may take a few moments."):
                        ingest_document(file_path)
                    st.success("✅ Document ingested successfully!")
                    st.session_state.rag_pipeline = None
                    st.session_state.vectorstore_ready = False
                except Exception as e:
                    st.error(f"Error ingesting document: {str(e)}")
            else:
                st.error("Invalid file format. Please provide a PDF file.")
        else:
            st.error("File not found. Please check the path.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with LangChain, Hugging Face, and Streamlit | "
    "Vector Store: Pinecone"
    "</div>",
    unsafe_allow_html=True
)
