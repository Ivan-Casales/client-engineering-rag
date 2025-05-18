import pytest
from app.services.utility.pdf_parser import extract_chunks_from_pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

class DummyDoc:
    def __init__(self, page_content):
        self.page_content = page_content

@pytest.fixture(autouse=True)
def patch_pdf_components(monkeypatch):
    # Bypass __init__ to avoid file existence checks
    def dummy_init(self, path, headers=None):
        return None
    monkeypatch.setattr(PyPDFLoader, "__init__", dummy_init)
    # Mock the PDF loader load method to return two dummy pages
    monkeypatch.setattr(PyPDFLoader, "load", lambda self: [DummyDoc("foo"), DummyDoc("bar")])
    # Mock the splitter to return docs unchanged
    monkeypatch.setattr(CharacterTextSplitter, "split_documents", lambda self, docs: docs)

@pytest.mark.parametrize("path,expected", [
    ("dummy.pdf", ["foo", "bar"]),
    ("another.pdf", ["foo", "bar"])
])
def test_extract_chunks_from_pdf(path, expected):
    chunks = extract_chunks_from_pdf(path)
    assert chunks == expected
