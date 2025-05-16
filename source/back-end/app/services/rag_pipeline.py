from app.services.watsonx_client import get_embedding, rerank_documents, generate_answer_with_context
from app.services.chroma_db import query_similar

def generate_answer(question: str) -> str:
    question_embedding = get_embedding(question)

    retrieval_results = query_similar(question_embedding, k=5)
    documents = retrieval_results.get("documents", [[]])[0]

    if not documents:
        return "No relevant documents found."

    top_documents = rerank_documents(question, documents)

    context = "\n\n".join(top_documents[:3])

    return generate_answer_with_context(context, question)