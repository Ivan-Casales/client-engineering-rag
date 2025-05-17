import pytest
from langchain.chains import RetrievalQA
from app.services.rag_pipeline import build_rag_chain, generate_answer

class DummyVectorStore:
    def __init__(self, retriever):
        self._retriever = retriever
    def as_retriever(self):
        return self._retriever

class DummyLLM:
    pass

class DummyChain:
    def __init__(self, response):
        self._response = response
    def run(self, question):
        return self._response

def test_build_rag_chain_invokes_from_chain_type(monkeypatch):
    dummy_retriever = object()
    vectorstore = DummyVectorStore(dummy_retriever)
    dummy_llm = DummyLLM()

    # Capture calls to from_chain_type
    called = {}
    def fake_from_chain_type(*, llm, retriever):
        called['llm'] = llm
        called['retriever'] = retriever
        return DummyChain("test_response")

    monkeypatch.setattr(RetrievalQA, 'from_chain_type', staticmethod(fake_from_chain_type))

    chain = build_rag_chain(vectorstore, dummy_llm)
    assert isinstance(chain, DummyChain)
    assert called['llm'] is dummy_llm
    assert called['retriever'] is dummy_retriever

@pytest.mark.parametrize("question,expected", [
    ("Hello?", "Answer1"),
    ("Bye?", "Answer2")
])
def test_generate_answer_returns_chain_response(question, expected):
    class FakeChain:
        def run(self, q):
            assert q == question
            return expected

    fake_chain = FakeChain()
    result = generate_answer(question, fake_chain)
    assert result == expected
