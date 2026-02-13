
from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os

from src.config import (
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    TOP_K
)
from src.retrieval import get_retriever

class RAGGenerator:
    """
    Handles Answer Generation using LLM and Retrieved Context.
    """
    def __init__(self):
        self.llm = self._initialize_llm()
        self.retriever = get_retriever()
        self.chain = self._build_chain()

    def _initialize_llm(self):
        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY is required. Please set it in your .env file.")
        
        print(f"Initializing LLM: {HUGGINGFACE_MODEL}")
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_MODEL,
            temperature=0.1,
            max_new_tokens=512,
            do_sample=True,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            # Add specific model params if needed for Llama 3
        )

    def _build_chain(self):
        # Strict Prompt for RAG
        template = """You are an intelligent assistant for finding information in documents.
Use the following pieces of retrieved context to answer the question.

Rules:
1. If the answer is not in the context, strictly say "I don't know based on the provided documents."
2. Do not hallucinate or use outside knowledge.
3. Cite the source and page number if available (e.g., [Source: doc.pdf, Page: 5]).

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            formatted = []
            for doc in docs:
                content = doc.page_content
                meta = doc.metadata
                source = meta.get('source', 'Unknown')
                page = meta.get('page', 'Unknown')
                formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
            return "\n\n".join(formatted)

        # Retrieval Chain
        # We wrap the retriever's get_relevant_documents method to fit Runnable interface if needed,
        # but since we defined a custom retriever class, let's wrap it in a lambda or RunnableLambda
        
        # Actually, our custom retriever returns a list of dicts. 
        # We need it to return Documents or string for the chain.
        # Let's adjust retrieval.py to return Documents or fix here.
        
        # FIX: The retrieval logic in src/retrieval.py returns dicts. 
        # The chain expects standard inputs.
        # Let's manually invoke retrieval instead of full LCEL for custom retriever simplicity.
        pass 

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Generate answer for a question.
        """
        # 1. Retrieve
        # Note: using the custom retriever directly
        retrieved_docs_dicts = self.retriever.get_relevant_documents(question)
        
        # Convert back to objects/format for prompt
        # We need to construct the context string
        context_str = ""
        source_docs = []
        
        for i, doc in enumerate(retrieved_docs_dicts):
             # Ensure we handle checking if it's a Document object or dict
             # In retrieval.py we returned dicts: {'page_content': ..., 'metadata': ...}
             content = doc.get('page_content', '')
             metadata = doc.get('metadata', {})
             
             source = metadata.get('source', 'Unknown')
             page = metadata.get('page', 'Unknown')
             
             context_str += f"Document {i+1} [Source: {source}, Page: {page}]:\n{content}\n\n"
             source_docs.append(doc)
             
        # 2. Generate
        # We use the LLM directly with the prompt
        
        full_prompt = f"""You are an intelligent assistant for finding information in documents.
Use the following pieces of retrieved context to answer the question.

Rules:
1. If the answer is not in the context, strictly say "I don't know based on the provided documents."
2. Do not hallucinate or use outside knowledge.
3. Cite the source and page number if available (e.g., [Source: doc.pdf, Page: 5]).

Context:
{context_str}

Question: {question}

Answer:"""

        if not context_str.strip():
             return {
                 "answer": "I couldn't find any relevant information in the documents to answer your question.",
                 "source_documents": []
             }

        response = self.llm.invoke(full_prompt)
        
        return {
            "answer": response,
            "source_documents": source_docs
        }

# Global instance
_generator_instance = None

def get_generator():
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = RAGGenerator()
    return _generator_instance
