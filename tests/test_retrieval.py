from langchain_core.documents import Document

from src.retrieval import AdvancedRetriever, SafeCrossEncoderReranker


class _ModelWithScore:
    def score(self, pairs):
        return [0.2, 0.9, 0.5]


class _FailingRetriever:
    def invoke(self, query: str):
        raise RuntimeError("primary retrieval failed")


class _FallbackRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query: str):
        return self.docs


def test_safe_cross_encoder_reranker_uses_score_and_orders_by_relevance():
    docs = [
        Document(page_content="doc-a", metadata={}),
        Document(page_content="doc-b", metadata={}),
        Document(page_content="doc-c", metadata={}),
    ]
    reranker = SafeCrossEncoderReranker(model=_ModelWithScore(), top_n=2)
    top_docs = reranker.compress_documents(docs, "query")

    assert len(top_docs) == 2
    assert top_docs[0].page_content == "doc-b"
    assert "score" in top_docs[0].metadata


def test_advanced_retriever_falls_back_to_base_retriever():
    fallback_docs = [Document(page_content="fallback", metadata={"source": "x.pdf"})]

    retriever = AdvancedRetriever.__new__(AdvancedRetriever)
    retriever.retrieve_chain = _FailingRetriever()
    retriever.base_retriever = _FallbackRetriever(fallback_docs)

    docs = retriever.get_relevant_documents("anything")
    assert len(docs) == 1
    assert docs[0].page_content == "fallback"
