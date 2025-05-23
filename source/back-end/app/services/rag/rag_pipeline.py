import re
from langchain.chains import RetrievalQA
from app.services.utility.prompt_templates import STRICT_CONTEXT_PROMPT
from .reranker import ReRanker
from app.services.container import llm
from app.services.utility.security import sanitize_input

def generate_answer(question: str, rag_chain: RetrievalQA, reranker: ReRanker, top_k: int = 5) -> str:
    question = sanitize_input(question)
    
    docs = rag_chain.retriever.get_relevant_documents(question)
    
    if not docs:
        return "I don't know. I couldn't find any information about that in the provided documents."

    reranked_docs = reranker.rerank_documents(question, docs, top_k=top_k)

    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    prompt_formatted = STRICT_CONTEXT_PROMPT.format(context=context, question=question)

    raw = llm(prompt_formatted).strip()

    cleaned = re.sub(r'(?:Question:?)+\s*$', '', raw, flags=re.IGNORECASE).strip()
    return cleaned
