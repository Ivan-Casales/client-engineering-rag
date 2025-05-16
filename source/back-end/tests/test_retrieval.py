import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.watsonx_client import get_embedding
from app.services.chroma_db import query_similar

def test_query_similar(sample_text: str, k: int = 3):
    print("TEST RETRIEVAL")
    try:
        emb = get_embedding(sample_text)
        print(f"Embedding length: {len(emb)}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return

    try:
        results = query_similar(emb, k=k)
        # New: imprimimos de forma legible
        docs_matrix = results.get("documents", [])
        docs = docs_matrix[0] if docs_matrix else []
        print(f"Top {k} similar documents to '{sample_text}':")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc[:100]}{'...' if len(doc)>100 else ''}")
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")

if __name__ == "__main__":
    sample = "¿Qué es Watsonx.ai?"
    test_query_similar(sample)
