import src.ingestion as ingestion


def test_ingest_documents_empty_list():
    result = ingestion.ingest_documents([])
    assert result == {"ingested": [], "failed": [], "total_chunks": 0}


def test_ingest_documents_source_name_length_mismatch():
    try:
        ingestion.ingest_documents(["a.pdf"], source_names=["a.pdf", "b.pdf"])
        assert False, "Expected ValueError for mismatched source_names length"
    except ValueError as exc:
        assert "source_names length must match file_paths length" in str(exc)


def test_ingest_documents_partial_failure(monkeypatch):
    class _FakeEmbeddings:
        pass

    class _FakeVectorStore:
        pass

    def _fake_ingest(file_path, embeddings, vectorstore, source_name=None):
        if "bad" in file_path:
            raise RuntimeError("boom")
        return {"pages": 2, "chunks": 5}

    monkeypatch.setattr(ingestion, "HuggingFaceEmbeddings", lambda model_name: _FakeEmbeddings())
    monkeypatch.setattr(ingestion, "get_chroma_vectorstore", lambda embeddings, allow_repair=True: _FakeVectorStore())
    monkeypatch.setattr(ingestion, "_ingest_document_with_resources", _fake_ingest)

    result = ingestion.ingest_documents(
        ["ok.pdf", "bad.pdf"],
        source_names=["ok.pdf", "bad.pdf"],
    )

    assert len(result["ingested"]) == 1
    assert len(result["failed"]) == 1
    assert result["total_chunks"] == 5
    assert result["ingested"][0]["source"] == "ok.pdf"
    assert result["failed"][0]["source"] == "bad.pdf"


def test_clear_vectorstore_deletes_all_ids(monkeypatch):
    class _FakeEmbeddings:
        pass

    class _FakeVectorStore:
        def __init__(self):
            self.deleted = None

        def get(self):
            return {"ids": ["1", "2", "3"]}

        def delete(self, ids=None):
            self.deleted = ids

    fake_vs = _FakeVectorStore()
    monkeypatch.setattr(ingestion, "HuggingFaceEmbeddings", lambda model_name: _FakeEmbeddings())
    monkeypatch.setattr(ingestion, "get_chroma_vectorstore", lambda embeddings, allow_repair=True: fake_vs)

    deleted = ingestion.clear_vectorstore()
    assert deleted == 3
    assert fake_vs.deleted == ["1", "2", "3"]
