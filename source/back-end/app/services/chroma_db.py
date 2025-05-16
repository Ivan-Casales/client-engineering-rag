import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

client = chromadb.Client(ChromaSettings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=settings.CHROMA_PERSIST_DIRECTORY
))
collection = client.get_or_create_collection(name="docs")

def add_document(id: str, text: str, embedding: list[float]):
    """
    Add a document to the 'docs' collection in ChromaDB.

    Parameters:
    - id (str): A unique identifier for the document.
    - text (str): The text content to be indexed.
    - embedding (list[float]): The embedding vector corresponding to the text.

    Raises:
    - ValueError: If id or text is empty, or embedding is not a list.
    - RuntimeError: If an error occurs while adding the document to ChromaDB.
    """
    if not id or not text or not isinstance(embedding, list):
        raise ValueError("id, text and embedding must be provided correctly")
    try:
        collection.add(
            documents=[text],
            ids=[id],
            embeddings=[embedding]
        )
    except Exception as e:
        raise RuntimeError(f"Error adding document to ChromaDB: {e}")

def query_similar(text_embedding: list[float], k: int = 3):
    """
    Retrieve the top-k documents most similar to the given embedding.

    Parameters:
    - text_embedding (list[float]): The embedding vector to query against.
    - k (int): The number of similar documents to return (default is 3).

    Returns:
    - dict: A dictionary containing 'documents', 'ids', and 'embeddings' matrices.

    Raises:
    - ValueError: If text_embedding is not a list.
    - RuntimeError: If an error occurs while querying ChromaDB.
    """
    if not isinstance(text_embedding, list):
        raise ValueError("text_embedding must be a list of floats")
    try:
        return collection.query(
            query_embeddings=[text_embedding],
            n_results=k
        )
    except Exception as e:
        raise RuntimeError(f"Error querying ChromaDB: {e}")