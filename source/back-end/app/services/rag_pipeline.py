from typing import List
from app.services.watsonx_client import (
    get_embedding,
    rerank_documents,
    generate_answer_with_context
)
from app.services.chroma_db import query_similar

def generate_answer(question: str, k: int = 5, top_k: int = 3) -> str:
    """
    Generates an answer to the specified question using a RAG pipeline:
      1. Compute embedding for the question.
      2. Retrieve the top-k similar documents from the vector database.
      3. Rerank the retrieved documents.
      4. Generate an answer based on the top_k documents.

    Parameters:
    - question (str): The user's question to be answered.
    - k (int): Number of documents to retrieve initially (default: 5).
    - top_k (int): Number of top documents to use for answer generation (default: 3).

    Returns:
    - str: The generated answer text.

    Raises:
    - RuntimeError: If embedding generation fails.
    - RuntimeError: If document retrieval fails.
    - RuntimeError: If answer generation fails.
    """
    try:
        question_embedding = get_embedding(question)
    except Exception as e:
        raise RuntimeError(f"Embedding error: {e}")

    try:
        results = query_similar(question_embedding, k=k)
        docs_matrix = results.get("documents", [])
        documents: List[str] = docs_matrix[0] if docs_matrix else []
    except Exception as e:
        raise RuntimeError(f"Retrieval error: {e}")

    if not documents:
        return "No relevant documents found."

    try:
        top_documents = rerank_documents(question, documents)
    except Exception:
        top_documents = documents

    context = "\n\n".join(top_documents[:top_k])

    try:
        answer = generate_answer_with_context(context, question)
    except Exception as e:
        raise RuntimeError(f"Answer generation error: {e}")

    return answer