import sys
import importlib
import pytest

# Dummy classes for patching
class DummyEmbeddings:
    pass

class DummyVectorStore:
    pass

class DummyLLM:
    pass

class DummyChain:
    pass

@pytest.fixture(autouse=True)
def patch_container_dependencies(monkeypatch):
    # Patch WatsonXEmbeddings and WatsonXLLM in watsonx_client module
    import app.services.watsonx_client as wxc
    monkeypatch.setattr(wxc, 'WatsonXEmbeddings', lambda: DummyEmbeddings())
    monkeypatch.setattr(wxc, 'WatsonXLLM', lambda *args, **kwargs: DummyLLM())

    # Patch load_vectorstore in chroma_db module
    import app.services.chroma_db as cdb
    monkeypatch.setattr(cdb, 'load_vectorstore', lambda embedding_model, persist_directory: DummyVectorStore())

    # Patch build_rag_chain in rag_pipeline module
    import app.services.rag_pipeline as rp
    monkeypatch.setattr(rp, 'build_rag_chain', lambda vs, llm: DummyChain())

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
