"""
Streamlit UI for document-based Q&A using RAG.
"""

import os
import tempfile

import streamlit as st

try:
    from src.config import HUGGINGFACE_MODEL, RERANK_TOP_K, TOP_K
    from src.generator import get_generator, reset_generator
    from src.ingestion import clear_vectorstore, ingest_documents
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
    :root {
        --brand: #0f4c5c;
        --brand-soft: #edf6f8;
        --accent: #2f8f9d;
        --ink: #102a43;
        --muted: #5c6f82;
        --surface: #f7fafc;
        --ok: #1f7a5a;
        --warn: #9f6a00;
    }
    .hero {
        background: linear-gradient(125deg, #f7fafc 0%, #e6f2f5 45%, #dbeef2 100%);
        border: 1px solid #d4e7ec;
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        margin: 0;
        color: var(--ink);
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
        font-size: 1rem;
    }
    .chip-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .chip {
        background: #ffffff;
        color: var(--brand);
        border: 1px solid #c8e0e6;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e6edf1;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .panel-title {
        margin: 0 0 0.4rem 0;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
    }
    .answer-box {
        background: #ffffff;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border-left: 4px solid var(--accent);
        border: 1px solid #dbe8ee;
        margin-top: 0.75rem;
        line-height: 1.65;
    }
    .metadata-box {
        font-size: 0.82rem;
        color: #4f6272;
        background: #f5f9fc;
        border: 1px solid #e2edf3;
        padding: 0.45rem 0.55rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .kpi-ok {
        color: var(--ok);
        font-weight: 700;
    }
    .kpi-warn {
        color: var(--warn);
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "rag_generator" not in st.session_state:
    st.session_state.rag_generator = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "latest_ingested_sources" not in st.session_state:
    st.session_state.latest_ingested_sources = []


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


st.markdown(
    """
    <div class="hero">
      <h1 class="hero-title">Advanced Document Q&A</h1>
      <p class="hero-subtitle">High-precision retrieval with semantic chunking, hybrid search, and reranking.</p>
      <div class="chip-row">
        <span class="chip">Vector Store: ChromaDB</span>
        <span class="chip">Model: Hugging Face</span>
        <span class="chip">Reranker Top-K: dynamic</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Runtime")
    st.metric("Retriever Top-K", TOP_K)
    st.metric("Rerank Top-K", RERANK_TOP_K)
    st.caption(f"LLM: `{HUGGINGFACE_MODEL}`")

    st.markdown("---")
    st.header("Status")
    if st.session_state.rag_generator is None:
        st.markdown('<span class="kpi-warn">Engine: Not initialized</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="kpi-ok">Engine: Ready</span>', unsafe_allow_html=True)
    st.caption(f"Saved Q&A turns: {len(st.session_state.qa_history)}")

    st.markdown("---")
    if st.button("Clear Q&A History"):
        st.session_state.qa_history = []
        st.success("History cleared.")

tab1, tab2 = st.tabs(["Chat & Query", "Document Knowledge Base"])

with tab1:
    left, right = st.columns([2.25, 1], gap="large")

    with left:
        st.markdown('<div class="panel"><p class="panel-title">Ask a Question</p></div>', unsafe_allow_html=True)

        if st.session_state.rag_generator is None:
            if st.button("Start RAG Engine", type="primary", use_container_width=True):
                initialize_system()

        if st.session_state.rag_generator is not None:
            use_latest_only = st.checkbox(
                "Use only latest uploaded files",
                value=True,
                help="When enabled, answers are restricted to files from the most recent successful ingestion.",
            )
            with st.form("qa_form"):
                question = st.text_input(
                    "Question",
                    placeholder="e.g., What are the key deliverables and deadlines in the document?",
                    key="question_input",
                )
                submit = st.form_submit_button("Generate Answer", type="primary", use_container_width=True)

            if submit:
                if question and question.strip():
                    try:
                        source_filter = None
                        if use_latest_only:
                            source_filter = st.session_state.latest_ingested_sources or None

                        with st.spinner("Retrieving, reranking, and generating..."):
                            result = st.session_state.rag_generator.answer_question(
                                question,
                                source_filter=source_filter,
                            )
                            answer = result.get("answer", "No answer generated.")
                            docs = result.get("source_documents", [])

                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                        st.session_state.qa_history.insert(
                            0,
                            {
                                "question": question.strip(),
                                "answer": answer,
                                "references": len(docs),
                            },
                        )
                        st.session_state.qa_history = st.session_state.qa_history[:10]

                        if docs:
                            st.markdown("### Supported Evidence")
                            for i, doc in enumerate(docs, 1):
                                content = doc.get("page_content", "No content")
                                metadata = doc.get("metadata", {})
                                source = metadata.get("source", "Unknown File")
                                page = metadata.get("page", "N/A")
                                score = metadata.get("score", None)

                                with st.expander(f"Reference {i}: {source} (Page {page})", expanded=(i == 1)):
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
            st.info("Initialize the engine to begin querying documents.")

    with right:
        st.markdown(
            """
            <div class="panel">
              <p class="panel-title">Query Tips</p>
              <ul>
                <li>Ask precise questions with nouns and dates.</li>
                <li>Request citations to verify evidence quickly.</li>
                <li>Break complex asks into smaller questions.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel">
              <p class="panel-title">Recent Questions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.qa_history:
            for item in st.session_state.qa_history[:5]:
                with st.expander(f"Q: {item['question'][:58]}"):
                    st.write(item["answer"])
                    st.caption(f"References used: {item['references']}")
        else:
            st.caption("No recent questions yet.")

with tab2:
    st.markdown('<div class="panel"><p class="panel-title">Batch Ingestion</p></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to be semantically chunked and indexed.",
    )

    if uploaded_files:
        total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        c1, c2 = st.columns(2)
        c1.metric("Selected Files", len(uploaded_files))
        c2.metric("Total Size (MB)", f"{total_size_mb:.2f}")

        with st.expander("Selected Files", expanded=True):
            for f in uploaded_files:
                st.caption(f"- {f.name}")

        replace_existing = st.checkbox(
            "Replace existing knowledge base before indexing",
            value=True,
            help="Recommended when you want answers only from newly uploaded documents.",
        )

        if st.button("Ingest & Index All", type="primary", use_container_width=True):
            tmp_paths = []
            source_names = []
            try:
                if replace_existing:
                    deleted_count = clear_vectorstore()
                    st.info(f"Cleared existing index entries: {deleted_count}")

                progress = st.progress(0.0, text="Preparing files...")
                total_files = max(len(uploaded_files), 1)
                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_paths.append(tmp_file.name)
                        source_names.append(uploaded_file.name)
                    progress.progress((idx + 1) / total_files, text=f"Prepared {idx + 1}/{len(uploaded_files)}")

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
                    with st.expander("Ingestion Details", expanded=True):
                        for row in ingested:
                            st.caption(f"OK: {row['source']} | pages={row['pages']} chunks={row['chunks']}")

                    st.session_state.latest_ingested_sources = [row["source"] for row in ingested]
                    reset_generator()
                    st.session_state.rag_generator = None
                    st.info("Engine reset to refresh retrieval index. Click Start RAG Engine before next query.")

                if failed:
                    st.warning(f"{len(failed)} file(s) failed.")
                    with st.expander("Failed Files", expanded=True):
                        for row in failed:
                            st.caption(f"FAIL: {row['source']} | {row['error']}")
            except Exception as e:
                st.error(f"Batch ingestion failed: {e}")
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
