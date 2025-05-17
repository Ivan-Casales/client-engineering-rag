from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.config import settings
from app.services import container

app = FastAPI(title="Watsonx RAG Assistant with LangChain")

app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """
    Perform startup checks and initialization logic.
    """
    if not settings.WATSONX_URL.startswith("https://"):
        raise RuntimeError("Invalid WATSONX_URL configuration")

    try:
        _ = container.rag_chain
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG chain: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions and return structured JSON errors.
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
