import pytest

from src.ingestion import ingest_document


def test_ingest_document_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_document("does-not-exist.pdf")


def test_ingest_document_rejects_non_pdf(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Only PDF files are supported"):
        ingest_document(str(file_path))
