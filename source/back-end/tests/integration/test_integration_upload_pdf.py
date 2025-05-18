import sys
from types import ModuleType, SimpleNamespace
import io
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from reportlab.pdfgen import canvas

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

loader_stub = ModuleType("app.services.vectorstore.loader_service")
loader_stub.process_pdf_upload = lambda file_bytes: (1, "")  # siempre 1 chunk, sin error
sys.modules["app.services.vectorstore.loader_service"] = loader_stub

chroma_stub = ModuleType("app.services.vectorstore.chroma_db")
chroma_stub.load_vectorstore = lambda *args, **kwargs: None
sys.modules["app.services.vectorstore.chroma_db"] = chroma_stub

rag_stub = ModuleType("app.services.rag.rag_pipeline")
rag_stub.generate_answer = lambda *args, **kwargs: "unused"
sys.modules["app.services.rag.rag_pipeline"] = rag_stub

chat_stub = ModuleType("app.services.rag.chat_service")
chat_stub.process_chat = lambda *args, **kwargs: ([], "")
sys.modules["app.services.rag.chat_service"] = chat_stub

for m in ("app.main", "app.api.routes"):
    sys.modules.pop(m, None)

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
        files = {"file": ("test.pdf", one_page_pdf_bytes, "application/pdf")}
        response = await client.post("/api/upload-pdf", files=files)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"detail": "1 chunks indexed successfully."}
