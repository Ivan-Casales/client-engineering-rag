import pytest
from app.services.loader_service import process_pdf_upload
from app.services.pdf_parser import extract_chunks_from_pdf
from app.services.chroma_db import load_vectorstore
from app.services.watsonx_client import WatsonXEmbeddings
from langchain.schema import Document

class DummyVectorStore:
    def __init__(self):
        self.added = []
    def add_documents(self, docs):
        self.added.extend(docs)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    # Stub extract_chunks_from_pdf
    monkeypatch.setattr(
        'app.services.loader_service.extract_chunks_from_pdf',
        lambda path: ['chunk1', 'chunk2']
    )
    # Stub WatsonXEmbeddings to avoid real API
    monkeypatch.setattr(
        'app.services.loader_service.WatsonXEmbeddings',
        lambda: object()
    )
    # Stub load_vectorstore to return a dummy store
    dummy_store = DummyVectorStore()
    monkeypatch.setattr(
        'app.services.loader_service.load_vectorstore',
        lambda emb, directory: dummy_store
    )
    return dummy_store

def test_process_pdf_upload_success(patch_dependencies):
    fake_bytes = b'%PDF-1.4 fake'
    count, error = process_pdf_upload(fake_bytes)

    assert count == 2
    assert error == ''

    # Verify documents passed to vectorstore
    added = patch_dependencies.added
    assert all(isinstance(doc, Document) for doc in added)
    assert [doc.page_content for doc in added] == ['chunk1', 'chunk2']


def test_process_pdf_upload_exception(monkeypatch):
    # Simulate parser failure
    monkeypatch.setattr(
        'app.services.loader_service.extract_chunks_from_pdf',
        lambda path: (_ for _ in ()).throw(ValueError('parse error'))
    )
    count, error = process_pdf_upload(b'data')

    assert count == 0
    assert 'parse error' in error