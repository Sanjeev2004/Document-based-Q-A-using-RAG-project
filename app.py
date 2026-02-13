"""
Minimal Streamlit UI for Document Based Q&A.
"""

import os
import tempfile

import streamlit as st

try:
    from src.generator import get_generator, reset_generator
    from src.ingestion import clear_vectorstore, ingest_documents
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()


st.set_page_config(page_title="Document Based Q&A", page_icon="Q&A", layout="wide")

if "rag_generator" not in st.session_state:
    st.session_state.rag_generator = None
if "latest_ingested_sources" not in st.session_state:
    st.session_state.latest_ingested_sources = []


def initialize_system() -> bool:
    try:
        with st.spinner("Starting engine..."):
            st.session_state.rag_generator = get_generator()
            return True
    except Exception as e:
        st.error(f"Engine error: {e}")
        return False


st.title("Document Based Q&A")

left, right = st.columns([1.8, 1], gap="large")

with left:
    st.subheader("Chat & Query")

    if st.session_state.rag_generator is None:
        if st.button("Start Engine", type="primary", use_container_width=True):
            initialize_system()
    else:
        st.caption("Engine ready")

    use_latest_only = st.checkbox("Use latest uploaded files only", value=True)

    with st.form("qa_form"):
        question = st.text_input("Question", placeholder="Ask from your document...")
        ask = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

    if ask:
        if not st.session_state.rag_generator:
            st.warning("Start the engine first.")
        elif not question or not question.strip():
            st.warning("Enter a question.")
        else:
            source_filter = None
            if use_latest_only:
                source_filter = st.session_state.latest_ingested_sources or None
            try:
                with st.spinner("Generating answer..."):
                    result = st.session_state.rag_generator.answer_question(
                        question.strip(), source_filter=source_filter
                    )
                answer = result.get("answer", "No answer generated.")
                docs = result.get("source_documents", [])
                st.markdown("**Answer**")
                st.write(answer)

                if docs:
                    st.markdown("**Evidence**")
                    for i, doc in enumerate(docs, 1):
                        metadata = doc.get("metadata", {})
                        source = metadata.get("source", "Unknown")
                        page = metadata.get("page", "N/A")
                        with st.expander(f"{i}. {source} (Page {page})", expanded=(i == 1)):
                            st.text(doc.get("page_content", "No content"))
            except Exception as e:
                st.error(f"Query failed: {e}")

with right:
    st.subheader("Upload")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    replace_existing = st.toggle("Replace old index", value=True)

    if st.button("Ingest Files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one PDF.")
        else:
            tmp_paths = []
            source_names = []
            try:
                if replace_existing:
                    deleted = clear_vectorstore()
                    st.caption(f"Cleared {deleted} existing chunks.")

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_paths.append(tmp_file.name)
                        source_names.append(uploaded_file.name)

                with st.spinner("Indexing files..."):
                    result = ingest_documents(tmp_paths, source_names=source_names)

                ingested = result.get("ingested", [])
                failed = result.get("failed", [])
                st.session_state.latest_ingested_sources = [r["source"] for r in ingested]
                reset_generator()
                st.session_state.rag_generator = None

                if ingested:
                    st.success(f"Ingested {len(ingested)} file(s). Start engine to query.")
                if failed:
                    st.warning(f"{len(failed)} file(s) failed.")
                    for row in failed:
                        st.caption(f"{row['source']}: {row['error']}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
