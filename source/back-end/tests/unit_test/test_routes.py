import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from app.main import app

@pytest.mark.asyncio
async def test_upload_pdf_success(mocker):
    # Mock file content
    fake_pdf_content = b"%PDF-1.4\n%Fake PDF content"

    # Mock the process_pdf_upload function
    mock_process = mocker.patch(
        "app.api.routes.process_pdf_upload",
        return_value=(5, "")
    )

    # Use ASGITransport to mount the FastAPI app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload-pdf",
            files={"file": ("fake.pdf", fake_pdf_content, "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["detail"] == "5 chunks indexed successfully."
    mock_process.assert_called_once()
