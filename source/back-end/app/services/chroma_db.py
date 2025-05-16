import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=None
))

collection = client.get_or_create_collection(name="docs")

def add_document(id: str, text: str, embedding: list[float]):
    collection.add(
        documents=[text],
        ids=[id],
        embeddings=[embedding]
    )

def query_similar(text_embedding: list[float], k=3):
    return collection.query(
        query_embeddings=[text_embedding],
        n_results=k
    )