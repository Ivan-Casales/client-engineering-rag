import sys
import importlib
import pytest

# Dummy classes for patching
class DummyEmbeddings:
    pass

class DummyVectorStore:
    def as_retriever(self, **kwargs):
        # Return a dummy retriever object
        return "dummy_retriever"

class DummyLLM:
    pass

class DummyChain:
    pass

@pytest.fixture(autouse=True)
def patch_container_dependencies(monkeypatch):
    # Patch WatsonXEmbeddings
    import app.services.watsonx.watsonx_embeddings as wxe
    monkeypatch.setattr(wxe, 'WatsonXEmbeddings', lambda: DummyEmbeddings())

    # Patch WatsonXLLM
    import app.services.watsonx.watsonx_llm as wxl
    monkeypatch.setattr(wxl, 'WatsonXLLM', lambda *args, **kwargs: DummyLLM())

    # Patch load_vectorstore in vectorstore.chroma_db
    import app.services.vectorstore.chroma_db as cdb
    monkeypatch.setattr(
        cdb,
        'load_vectorstore',
        lambda embedding_model, persist_directory: DummyVectorStore()
    )

    # Patch RetrievalQA.from_chain_type to return DummyChain
    from langchain.chains import RetrievalQA
    monkeypatch.setattr(
        RetrievalQA,
        'from_chain_type',
        staticmethod(lambda *, llm, retriever, chain_type_kwargs=None: DummyChain())
    )

    # Ensure container module is reloaded fresh
    if 'app.services.container' in sys.modules:
        del sys.modules['app.services.container']

    yield


def test_container_initialization():
    # Import container after all patches applied
    container = importlib.import_module('app.services.container')

    # Assertions: container attributes should be dummy instances
    assert isinstance(container.embedding_model, DummyEmbeddings)
    assert isinstance(container.vectorstore, DummyVectorStore)
    assert isinstance(container.llm, DummyLLM)
    assert isinstance(container.rag_chain, DummyChain)
