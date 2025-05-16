from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.config import settings

app = FastAPI(title="Watsonx RAG Assistant")

app.include_router(router, prefix="/api")

@app.on_event("startup")
async def validate_config():
    if not settings.WATSONX_URL.startswith("https://"):
        raise RuntimeError("Invalid WATSONX_URL configuration")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})