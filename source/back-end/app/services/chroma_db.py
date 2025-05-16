import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

client = chromadb.Client(ChromaSettings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=settings.CHROMA_PERSIST_DIRECTORY
))
collection = client.get_or_create_collection(name="docs")

def add_document(id: str, text: str, embedding: list[float]):
    if not id or not text or not isinstance(embedding, list):
        raise ValueError("id, text y embedding deben proporcionarse correctamente")
    try:
        collection.add(
            documents=[text],
            ids=[id],
            embeddings=[embedding]
        )
    except Exception as e:
        raise RuntimeError(f"Error al añadir documento a ChromaDB: {e}")

def query_similar(text_embedding: list[float], k: int = 3):
    if not isinstance(text_embedding, list):
        raise ValueError("text_embedding debe ser una lista de floats")
    try:
        return collection.query(
            query_embeddings=[text_embedding],
            n_results=k
        )
    except Exception as e:
        raise RuntimeError(f"Error al consultar ChromaDB: {e}")