import sys
from types import ModuleType, SimpleNamespace
import io
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from reportlab.pdfgen import canvas

# 1) Stub de app.core.config.settings para el arranque
config_stub = ModuleType("app.core.config")
config_stub.settings = SimpleNamespace(
    WATSONX_URL="https://dummy",
    WATSONX_APIKEY="dummyapikeystring",
    WATSONX_PROJECT_ID="dummyproj",
    CHROMA_PERSIST_DIRECTORY=".",
    MODEL_ID="dummy-model",
    EMBEDDING_MODEL_ID="dummy-embed-model",
    TEMPERATURE=0.0,
    MAX_NEW_TOKENS=1
)
sys.modules["app.core.config"] = config_stub

# 2) Stubs de los módulos externos
# 2a) app.services.watsonx_client
watsonx_stub = ModuleType("app.services.watsonx_client")
watsonx_stub.WatsonXEmbeddings = lambda: None
watsonx_stub.WatsonXLLM = lambda *args, **kwargs: None
sys.modules["app.services.watsonx_client"] = watsonx_stub

# 2b) app.services.chroma_db
class DummyVectorStore:
    def __init__(self):
        self.docs = []
    def add_documents(self, docs):
        self.docs.extend(docs)

chroma_stub = ModuleType("app.services.chroma_db")
chroma_stub.load_vectorstore = lambda embedding_model, persist_directory: DummyVectorStore()
sys.modules["app.services.chroma_db"] = chroma_stub

# 2c) app.services.rag_pipeline (no se usa aquí, pero prevenimos side-effects)
rag_stub = ModuleType("app.services.rag_pipeline")
rag_stub.build_rag_chain = lambda vs, llm: None
rag_stub.generate_answer = lambda question, chain: "dummy-answer"
sys.modules["app.services.rag_pipeline"] = rag_stub

# ——————>  Limpieza de caché para forzar recarga <——————
for module in (
    "app.main",
    "app.api.routes",
    "app.services.loader_service",
):
    sys.modules.pop(module, None)

# 3) Ahora importamos la app, que tomará nuestros stubs al cargar
from app.main import app


@pytest.fixture
def one_page_pdf_bytes():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(50, 100, "Hello world")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_pdf_integration(one_page_pdf_bytes):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {
            "file": (
                "test.pdf",
                one_page_pdf_bytes,
                "application/pdf"
            )
        }
        response = await client.post("/api/upload-pdf", files=files)

    assert response.status_code == status.HTTP_200_OK
    assert "1 chunks indexed successfully." in response.json()["detail"]
