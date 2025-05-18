import pytest
from tests.test_utility.utils_watsonx import rerank_documents

@ pytest.mark.integration
def test_rerank_documents_order():
    question = "What is Watsonx.ai?"
    docs = [
        "Watsonx.ai is IBM's AI platform.",
        "OpenAI develops GPT models.",
        "Watsonx.ai offers enterprise AI tools."
    ]
    ranked = rerank_documents(question, docs)
    # Expect IBM-related docs first
    assert ranked[0].startswith("Watsonx.ai"), "Most relevant document should mention Watsonx"
    assert ranked[-1].startswith("OpenAI"), "Least relevant should be the OpenAI doc"