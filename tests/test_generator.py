from langchain_core.documents import Document

from src.generator import RAGGenerator


class _FakeLLM:
    def invoke(self, prompt: str) -> str:
        return "mocked-answer"


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def get_relevant_documents(self, query: str):
        return self._docs


def _build_generator(docs):
    generator = RAGGenerator.__new__(RAGGenerator)
    generator.llm = _FakeLLM()
    generator.retriever = _FakeRetriever(docs)
    return generator


def test_answer_question_rejects_empty_input():
    generator = _build_generator([])
    result = generator.answer_question("   ")
    assert result["source_documents"] == []
    assert "non-empty question" in result["answer"]


def test_answer_question_returns_structured_response():
    docs = [
        Document(
            page_content="The contract duration is 12 months.",
            metadata={"source": "contract.pdf", "page": 2},
        )
    ]
    generator = _build_generator(docs)
    result = generator.answer_question("What is the duration?")

    assert result["answer"] == "mocked-answer"
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0]["metadata"]["source"] == "contract.pdf"
