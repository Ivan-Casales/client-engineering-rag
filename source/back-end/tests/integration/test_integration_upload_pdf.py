import sys
from types import ModuleType, SimpleNamespace
import io
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

# 1) Stub of app.core.config.settings for startup
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

# 2) Stubs of external modules to prevent real initialization
# 2a) app.services.watsonx_client stub
watsonx_stub = ModuleType("app.services.watsonx_client")
watsonx_stub.WatsonXEmbeddings = lambda: None
watsonx_stub.WatsonXLLM = lambda *args, **kwargs: None
sys.modules["app.services.watsonx_client"] = watsonx_stub

# 2b) app.services.chroma_db stub
class DummyVectorStore:
    def __init__(self):
        self.docs = []
    def add_documents(self, docs):
        self.docs.extend(docs)

chroma_stub = ModuleType("app.services.chroma_db")
chroma_stub.load_vectorstore = lambda embedding_model, persist_directory: DummyVectorStore()
sys.modules["app.services.chroma_db"] = chroma_stub

# 2c) app.services.rag_pipeline stub (with both build_rag_chain and generate_answer)
rag_stub = ModuleType("app.services.rag_pipeline")
rag_stub.build_rag_chain = lambda vs, llm: None
rag_stub.generate_answer = lambda question, chain: "dummy-answer"
sys.modules["app.services.rag_pipeline"] = rag_stub

# 3) Import the app with stubs in place
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
    # Mount the FastAPI app without a live server
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

    # Should return 200 OK and indicate at least 1 chunk indexed
    assert response.status_code == status.HTTP_200_OK
    assert "1 chunks indexed successfully." in response.json()["detail"]
