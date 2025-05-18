import pytest
from tests.test_utility.utils_watsonx import get_embedding

@ pytest.mark.integration
def test_get_embedding_returns_vector():
    # This will call the real WatsonXEmbeddings with live credentials
    emb = get_embedding("What is Watsonx.ai?")
    assert isinstance(emb, list), "Embedding should be a list of floats"
    assert len(emb) > 0, "Embedding vector should not be empty"
    assert all(isinstance(x, float) for x in emb)