from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Watsonx RAG Assistant")

app.include_router(router, prefix="/api")
