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

# 2) Stub the entire container, para que app.main.use ese stub en lugar de container.py real
container_stub = ModuleType("app.services.container")
container_stub.rag_chain   = "dummy_chain"
container_stub.reranker    = lambda *args, **kwargs: None
sys.modules["app.services.container"] = container_stub

# 3) Stub el pipeline de RAG: routes.ask_question importa generate_answer de aquí
rag_stub = ModuleType("app.services.rag.rag_pipeline")
rag_stub.generate_answer = lambda question, chain, reranker: f"Answer to {question}"
sys.modules["app.services.rag.rag_pipeline"] = rag_stub

# 4) Stub loader_service (no se usa en /ask, pero routes importa el módulo)
loader_stub = ModuleType("app.services.vectorstore.loader_service")
loader_stub.process_pdf_upload = lambda file_bytes: (0, "")
sys.modules["app.services.vectorstore.loader_service"] = loader_stub

# 5) Stub chat_service (igual, para que import routes no rompa)
chat_stub = ModuleType("app.services.rag.chat_service")
chat_stub.process_chat = lambda message, history: (history, "")
sys.modules["app.services.rag.chat_service"] = chat_stub

# Ahora importamos la app ya con todo stubbeado
from app.main import app

@pytest.mark.asyncio
async def test_ask_endpoint_returns_answer():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/ask", json={"question": "What is 2+2?"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"answer": "Answer to What is 2+2?"}
