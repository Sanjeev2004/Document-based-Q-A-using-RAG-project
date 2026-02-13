
from typing import Any, List, Optional, Set
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import os
from src.config import (
    EMBEDDING_MODEL,
    TOP_K,
    RERANK_TOP_K
)
from src.vectorstore import get_chroma_vectorstore

# --- SAFE INLINE IMPLEMENTATIONS ---

class SimpleEnsembleRetriever(BaseRetriever):
    """
    Inline implementation of EnsembleRetriever.
    """
    retrievers: List[BaseRetriever]
    weights: Optional[List[float]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_docs_lists = [retriever.invoke(query) for retriever in self.retrievers]
        
        combined_docs = []
        seen_content = set()
        max_len = max(len(l) for l in all_docs_lists)
        for i in range(max_len):
            for j, doc_list in enumerate(all_docs_lists):
                if i < len(doc_list):
                    doc = doc_list[i]
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        doc.metadata["retriever_source"] = f"retriever_{j}"
                        combined_docs.append(doc)
        return combined_docs


class SafeCrossEncoderReranker:
    """
    Inline implementation of CrossEncoderReranker that checks for .score() vs .predict().
    """
    def __init__(self, model, top_n=3):
        self.model = model
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents: return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        scores = []
        try:
            # 1. Try .score() (LangChain standard)
            if hasattr(self.model, 'score'):
                scores = self.model.score(pairs)
            # 2. Try .predict() (SentenceTransformers standard)
            elif hasattr(self.model, 'predict'):
                scores = self.model.predict(pairs)
            else:
                print(f"[Reranker Warning] Model {type(self.model)} has neither score() nor predict().")
                return documents[:self.top_n]
        except Exception as e:
             print(f"[Reranker Error] Scoring failed: {e}")
             return documents[:self.top_n]

        # Combine and Sort
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Attach score to metadata
        top_docs = []
        for doc, score in doc_score_pairs[:self.top_n]:
            doc.metadata['score'] = float(score)
            top_docs.append(doc)
            
        return top_docs


class SafeContextualCompressionRetriever(BaseRetriever):
    """
    Inline wrapper for compression to avoid import errors.
    """
    base_compressor: Any
    base_retriever: Any
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        return self.base_compressor.compress_documents(docs, query)


class AdvancedRetriever:
    """
    Handles Hybrid Search (BM25 + Vector) and Reranking using ChromaDB.
    """
    def __init__(self):
        print("Initializing AdvancedRetriever...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = get_chroma_vectorstore(self.embeddings, allow_repair=True)
        self.retrieve_chain = None
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        # 1. Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        # 2. BM25 Retriever
        bm25_retriever = None
        try:
            collection_data = self.vectorstore.get() 
            texts = collection_data['documents']
            metadatas = collection_data['metadatas']
            
            if texts:
                print(f"Initializing BM25 Retriever with {len(texts)} documents.")
                docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = TOP_K
            else:
                print("ChromaDB is empty. BM25 Retriever skipped.")
        except Exception as e:
            print(f"Error initializing BM25: {e}")

        # 3. Hybrid Base
        if bm25_retriever:
            print("Using SimpleEnsembleRetriever (Hybrid).")
            self.base_retriever = SimpleEnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]
            )
        else:
            print("Using Vector-Only Retriever.")
            self.base_retriever = vector_retriever

        # 4. Reranker
        print("Initializing SafeCrossEncoderReranker.")
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # ALWAYS use our safe inline class to guarantee behavior
        compressor = SafeCrossEncoderReranker(model=model, top_n=RERANK_TOP_K)
        
        # Wrap in our safe retriever
        self.retrieve_chain = SafeContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )

    def _filter_by_sources(self, docs: List[Document], source_filter: Optional[List[str]]) -> List[Document]:
        if not source_filter:
            return docs
        allowed: Set[str] = {s for s in source_filter if isinstance(s, str) and s.strip()}
        if not allowed:
            return docs
        return [d for d in docs if (d.metadata or {}).get("source") in allowed]

    def get_relevant_documents(self, query: str, source_filter: Optional[List[str]] = None) -> List[Document]:
        """
        Retrieve relevant documents with reranking and fallback.
        """
        print(f"Retrieving for query: {query}")
        try:
            # Try full chain
            docs = self.retrieve_chain.invoke(query)
        except Exception as e:
            print(f"!!! Error in Reranking Chain: {e}")
            try:
                # Fallback to base
                docs = self.base_retriever.invoke(query)
            except Exception as e2:
                print(f"!!! Error in Base Retrieval: {e2}")
                return []
        return self._filter_by_sources(docs, source_filter)

# Global instance
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = AdvancedRetriever()
    return _retriever_instance


def reset_retriever():
    global _retriever_instance
    _retriever_instance = None
