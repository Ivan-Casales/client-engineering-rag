import sys
from types import ModuleType, SimpleNamespace
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

# 1) Stub out settings so startup won’t fail
config_stub = ModuleType("app.core.config")
config_stub.settings = SimpleNamespace(
    WATSONX_URL="https://dummy",
    WATSONX_APIKEY="dummyapikey",
    WATSONX_PROJECT_ID="dummyproj",
    CHROMA_PERSIST_DIRECTORY=".",
    MODEL_ID="dummy-model",
    EMBEDDING_MODEL_ID="dummy-embed-model",
    TEMPERATURE=0.0,
    MAX_NEW_TOKENS=1
)
sys.modules["app.core.config"] = config_stub

# 2) Stub the WatsonX client module
watsonx_stub = ModuleType("app.services.watsonx_client")
watsonx_stub.WatsonXEmbeddings = lambda: None
watsonx_stub.WatsonXLLM = lambda *args, **kwargs: None
sys.modules["app.services.watsonx_client"] = watsonx_stub

# 3) Stub the Chroma DB module (accept any signature)
chroma_stub = ModuleType("app.services.chroma_db")
chroma_stub.load_vectorstore = lambda *args, **kwargs: None
sys.modules["app.services.chroma_db"] = chroma_stub

# 4) Stub the RAG pipeline module
rag_stub = ModuleType("app.services.rag_pipeline")
rag_stub.build_rag_chain = lambda vs, llm: "dummy_chain"
rag_stub.generate_answer = lambda question, chain: f"Answer to {question}"
sys.modules["app.services.rag_pipeline"] = rag_stub

# 5) Now import the FastAPI app with all stubs in place
from app.main import app

@pytest.mark.asyncio
async def test_ask_endpoint_returns_answer():
    # Mount FastAPI app without running a live server
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"question": "What is 2+2?"}
        response = await client.post("/api/ask", json=payload)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"answer": "Answer to What is 2+2?"}
