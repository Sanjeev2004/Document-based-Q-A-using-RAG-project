"""
Streamlit UI for Document-based Q&A using RAG (Advanced)
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

# Import from new src structure
try:
    from src.generator import get_generator
    from src.ingestion import ingest_document
    from src.config import PINECONE_INDEX_NAME
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced Document Q&A",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metadata-box {
        font-size: 0.85rem;
        color: #666;
        background: #f0f0f0;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_generator' not in st.session_state:
    st.session_state.rag_generator = None

def initialize_system():
    """Initialize the RAG generator."""
    try:
        with st.spinner("Initializing Advanced RAG System (Llama-3 + Reranking)..."):
            generator = get_generator()
            st.session_state.rag_generator = generator
            return True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.info("Please ensure you have set your API keys in .env")
        return False

# Main UI
st.markdown('<div class="main-header">🧠 Advanced Document Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Llama-3, Semantic Chunking & Cross-Encoder Reranking</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.success(f"Vector Index: {PINECONE_INDEX_NAME}")
    st.info("Model: Llama-3-70b (via Hugging Face)")
    st.info("Reranker: MS-MARCO MiniLM")
    
    st.markdown("---")
    st.header("Features")
    st.markdown("""
    - **Semantic Chunking**: Smarter context splitting
    - **Hybrid Retrieval**: Vector + Reranking validation
    - **Citation Mode**: Strict source referencing
    """)

# Tabs
tab1, tab2 = st.tabs(["💬 Chat & Query", "📂 Document Knowledge Base"])

# Tab 1: Question Answering
with tab1:
    st.header("Ask Intelligence")
    
    # Auto-initialize if needed
    if st.session_state.rag_generator is None:
         if st.button("Start RAG Engine", type="primary"):
             initialize_system()
    
    if st.session_state.rag_generator is not None:
        question = st.text_input(
            "What would you like to know?",
            placeholder="e.g., specific details from the uploaded contracts...",
            key="question_input"
        )
        
        if st.button("Generate Answer", type="primary") or question:
            if question:
                try:
                    with st.spinner("Retrieving, Reranking & Generating..."):
                        # Ensure we get a dict response
                        result = st.session_state.rag_generator.answer_question(question)
                        
                        # Handle potential raw string return from some legacy chains vs dict
                        if isinstance(result, str):
                            answer = result
                            docs = []
                        else:
                            answer = result.get("answer", "No answer generated.")
                            docs = result.get("source_documents", [])
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                        
                        # Display source documents
                        if docs:
                            st.markdown("---")
                            st.markdown("### 📚 Supported Evidence (Reranked)")
                            
                            for i, doc in enumerate(docs, 1):
                                content = doc.get('page_content', 'No content')
                                metadata = doc.get('metadata', {})
                                source = metadata.get('source', 'Unknown File')
                                page = metadata.get('page', 'N/A')
                                score = metadata.get('score', 'N/A') # If available from reranker
                                
                                with st.expander(f"Reference {i}: {source} (Page {page})"):
                                    st.markdown(f"**Relevance Context:**")
                                    st.text(content)
                                    st.markdown(f'<div class="metadata-box">Source: {source} | Page: {page}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("System initializing or waiting for start...")

# Tab 2: Document Ingestion
with tab2:
    st.header("Add Knowledge")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF to be semantically chunked and indexed."
    )
    
    if uploaded_file is not None:
        st.success(f"Selected: {uploaded_file.name}")
        
        if st.button("Ingest & Index", type="primary"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                with st.spinner("Processing: Parsing -> Semantic Chunking -> Embedding -> Indexing..."):
                    # Call the new ingestion logic
                    ingest_document(tmp_path)
                
                os.unlink(tmp_path)
                
                st.success("✅ Ingestion Complete! The system is now aware of this document.")
                # Clear generator to force context refresh if needed (though vector store is external)
                
            except Exception as e:
                st.error(f"Ingestion Failed: {str(e)}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
