
from typing import Dict, Any
from huggingface_hub import InferenceClient

from src.config import (
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL
)
from src.retrieval import get_retriever, reset_retriever

class RAGGenerator:
    """
    Handles Answer Generation using LLM and Retrieved Context.
    """
    def __init__(self):
        self.llm = self._initialize_llm()
        self.retriever = get_retriever()

    def _initialize_llm(self):
        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY is required. Please set it in your .env file.")
        
        print(f"Initializing LLM: {HUGGINGFACE_MODEL}")
        return InferenceClient(api_key=HUGGINGFACE_API_KEY)

    def _generate_text(self, prompt: str) -> str:
        if hasattr(self.llm, "chat_completion"):
            response = self.llm.chat_completion(
                model=HUGGINGFACE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            text = response.choices[0].message.content if response and response.choices else ""
            return text or "I couldn't generate an answer at the moment."

        if hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
            return response if isinstance(response, str) else str(response)

        return "I couldn't generate an answer at the moment."

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Generate answer for a question.
        """
        if not isinstance(question, str) or not question.strip():
            return {
                "answer": "Please enter a non-empty question.",
                "source_documents": []
            }

        question = question.strip()

        # 1. Retrieve
        retrieved_docs = self.retriever.get_relevant_documents(question)

        context_str = ""
        source_docs = []

        for i, doc in enumerate(retrieved_docs):
            content = getattr(doc, "page_content", "") or ""
            metadata = getattr(doc, "metadata", {}) or {}

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            context_str += f"Document {i + 1} [Source: {source}, Page: {page}]:\n{content}\n\n"
            source_docs.append({
                "page_content": content,
                "metadata": metadata
            })

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
                "answer": "I couldn't find relevant information in the indexed documents.",
                "source_documents": []
            }

        answer_text = self._generate_text(full_prompt)

        return {
            "answer": answer_text,
            "source_documents": source_docs
        }

# Global instance
_generator_instance = None

def get_generator():
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = RAGGenerator()
    return _generator_instance


def reset_generator():
    global _generator_instance
    _generator_instance = None
    reset_retriever()
