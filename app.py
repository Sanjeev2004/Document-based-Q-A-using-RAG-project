"""
Streamlit UI for document-based Q&A using RAG.
"""

import os
import tempfile

import streamlit as st

try:
    from src.generator import get_generator, reset_generator
    from src.ingestion import ingest_documents
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()


st.set_page_config(
    page_title="Advanced Document Q&A",
    page_icon="Q&A",
    layout="wide",
)

st.markdown(
    """
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    """,
    unsafe_allow_html=True,
)

if "rag_generator" not in st.session_state:
    st.session_state.rag_generator = None


def initialize_system() -> bool:
    """Initialize RAG runtime."""
    try:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_generator = get_generator()
            return True
    except Exception as e:
        msg = str(e)
        st.error(f"Error initializing system: {msg}")
        if "default_tenant" in msg.lower() or "range start index" in msg.lower():
            st.info(
                "ChromaDB appears corrupted/incompatible. The app can auto-repair by recreating "
                "the local vector store directory and re-ingesting documents."
            )
        else:
            st.info("Please confirm your API keys are set in `.env`.")
        return False


st.markdown('<div class="main-header">Advanced Document Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by Llama-3, Semantic Chunking, and Cross-Encoder Reranking</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System Status")
    st.success("Vector Store: ChromaDB (Local)")
    st.info("Model: Llama-3-70b (via Hugging Face)")
    st.info("Reranker: MS-MARCO MiniLM")

    st.markdown("---")
    st.header("Features")
    st.markdown(
        """
    - **Semantic Chunking**: Smarter context splitting
    - **Hybrid Retrieval**: Vector + reranking validation
    - **Citation Mode**: Strict source referencing
    """
    )

tab1, tab2 = st.tabs(["Chat & Query", "Document Knowledge Base"])

with tab1:
    st.header("Ask Intelligence")

    if st.session_state.rag_generator is None:
        if st.button("Start RAG Engine", type="primary"):
            initialize_system()

    if st.session_state.rag_generator is not None:
        with st.form("qa_form"):
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., specific details from the uploaded contracts...",
                key="question_input",
            )
            submit = st.form_submit_button("Generate Answer", type="primary")

        if submit:
            if question and question.strip():
                try:
                    with st.spinner("Retrieving, reranking, and generating..."):
                        result = st.session_state.rag_generator.answer_question(question)
                        answer = result.get("answer", "No answer generated.")
                        docs = result.get("source_documents", [])

                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    if docs:
                        st.markdown("---")
                        st.markdown("### Supported Evidence (Reranked)")

                        for i, doc in enumerate(docs, 1):
                            content = doc.get("page_content", "No content")
                            metadata = doc.get("metadata", {})
                            source = metadata.get("source", "Unknown File")
                            page = metadata.get("page", "N/A")
                            score = metadata.get("score", None)

                            with st.expander(f"Reference {i}: {source} (Page {page})"):
                                st.markdown("**Relevance Context:**")
                                st.text(content)
                                if score is not None:
                                    st.markdown(
                                        f'<div class="metadata-box">Source: {source} | Page: {page} | Score: {score:.4f}</div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="metadata-box">Source: {source} | Page: {page}</div>',
                                        unsafe_allow_html=True,
                                    )
                except Exception as e:
                    st.error(f"Error during generation: {e}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("System initializing or waiting for start...")

with tab2:
    st.header("Add Knowledge")

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload a PDF to be semantically chunked and indexed.",
    )

    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} file(s).")
        for f in uploaded_files:
            st.caption(f"- {f.name}")

        if st.button("Ingest & Index All", type="primary"):
            tmp_paths = []
            source_names = []
            try:
                progress = st.progress(0.0, text="Preparing files...")
                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_paths.append(tmp_file.name)
                        source_names.append(uploaded_file.name)
                    progress.progress((idx + 1) / max(len(uploaded_files), 1), text=f"Prepared {idx + 1}/{len(uploaded_files)}")

                with st.spinner("Batch processing: parsing, chunking, embedding, indexing..."):
                    result = ingest_documents(tmp_paths, source_names=source_names)

                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                ingested = result.get("ingested", [])
                failed = result.get("failed", [])
                total_chunks = result.get("total_chunks", 0)

                if ingested:
                    st.success(f"Ingested {len(ingested)} file(s), total chunks: {total_chunks}.")
                    for row in ingested:
                        st.caption(f"OK: {row['source']} | pages={row['pages']} chunks={row['chunks']}")

                    reset_generator()
                    st.session_state.rag_generator = None

                if failed:
                    st.warning(f"{len(failed)} file(s) failed.")
                    for row in failed:
                        st.caption(f"FAIL: {row['source']} | {row['error']}")
            except Exception as e:
                st.error(f"Batch ingestion failed: {e}")
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
