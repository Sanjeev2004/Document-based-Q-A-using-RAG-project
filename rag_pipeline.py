"""
RAG Pipeline: Retrieval-Augmented Generation
Handles question answering using retrieved context from Pinecone vector store
"""

import os
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone

from config import (
    EMBEDDING_MODEL,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    TOP_K
)


class RAGPipeline:
    """RAG Pipeline for document-based question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline with embeddings, vector store, and LLM."""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components of the RAG pipeline."""
        # Initialize embeddings
        print(f"Initializing embeddings: {EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load Pinecone vector store
        self._load_pinecone_vectorstore()
        
        # Initialize LLM
        print(f"Initializing LLM: {HUGGINGFACE_MODEL}...")
        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY is required. Please set it in your .env file.")
        
        llm = HuggingFaceHub(
            repo_id=HUGGINGFACE_MODEL,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 512,
                "do_sample": True
            },
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )
        
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use only the information provided in the context to answer the question.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("RAG pipeline initialized successfully!")
    
    def _load_pinecone_vectorstore(self):
        """Load Pinecone vector store."""
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file.")
        if not PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_ENVIRONMENT is required. Please set it in your .env file.")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        
        print(f"Loading Pinecone vector store: {PINECONE_INDEX_NAME}...")
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
        print("Pinecone vector store loaded successfully!")
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise RuntimeError("RAG pipeline not initialized. Call _initialize() first.")
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "source_documents": []
            }
    
    def get_relevant_chunks(self, question: str, k: int = TOP_K) -> List[Dict]:
        """
        Retrieve relevant chunks for a question without generating an answer.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized.")
        
        docs = self.vectorstore.similarity_search(question, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]


# Global instance (lazy initialization)
_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the global RAG pipeline instance.
    
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    
    return _pipeline_instance
