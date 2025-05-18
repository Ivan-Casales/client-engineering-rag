import pytest
from langchain.chains import RetrievalQA
from app.services.rag.rag_pipeline import generate_answer
from tests.test_utility.prompt_test import TEST_PROMPT
# Helper para construir el RAG chain
def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": TEST_PROMPT},
    )

class DummyVectorStore:
    def __init__(self, retriever):
        self._retriever = retriever

    def as_retriever(self, **kwargs):
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

    called = {}
    def fake_from_chain_type(*, llm, retriever, chain_type_kwargs=None):
        called['llm'] = llm
        called['retriever'] = retriever
        return DummyChain("test_response")

    monkeypatch.setattr(RetrievalQA, 'from_chain_type', staticmethod(fake_from_chain_type))
    chain = build_rag_chain(vectorstore, dummy_llm)

    assert isinstance(chain, DummyChain)
    assert called['llm'] is dummy_llm
    assert called['retriever'] is dummy_retriever

@pytest.mark.parametrize("question,expected", [
    ("Hello?", "I couldn't find"),
    ("Bye?", "Bye")
])
def test_generate_answer_returns_chain_response(question, expected):
    # Documento simulado con page_content
    class DummyDoc:
        def __init__(self, content):
            self.page_content = content

    class DummyRetriever:
        def get_relevant_documents(self, q):
            assert q == question
            # Ahora devolvemos objetos que tengan .page_content
            return [DummyDoc("doc1"), DummyDoc("doc2"), DummyDoc("doc3")]

    class FakeChain:
        def __init__(self):
            self.retriever = DummyRetriever()

        def run(self, q):
            assert q == question
            return expected

    class DummyReranker:
        def rerank_documents(self, question_param, context, top_k):
            # context es lista de DummyDoc
            assert question_param == question
            assert [d.page_content for d in context] == ["doc1", "doc2", "doc3"]
            assert top_k == 5
            return context

    fake_chain = FakeChain()
    reranker = DummyReranker()

    result = generate_answer(question, fake_chain, reranker)
    assert expected.lower() in result.lower()
