from typing import List
from app.services.watsonx_client import (
    get_embedding,
    rerank_documents,
    generate_answer_with_context
)
from app.services.chroma_db import query_similar

def generate_answer(question: str, k: int = 5, top_k: int = 3) -> str:
    try:
        question_embedding = get_embedding(question)
    except Exception as e:
        raise RuntimeError(f"Error de embedding: {e}")

    try:
        results = query_similar(question_embedding, k=k)
        docs_matrix = results.get("documents", [])
        documents: List[str] = docs_matrix[0] if docs_matrix else []
    except Exception as e:
        raise RuntimeError(f"Error en recuperación: {e}")

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
        raise RuntimeError(f"Error en generación de QA: {e}")

    return answer