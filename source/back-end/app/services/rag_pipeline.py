import re
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.llms.base import LLM
from .utility.prompt_templates import STRICT_CONTEXT_PROMPT

def build_rag_chain(vectorstore: Chroma, llm: LLM) -> RetrievalQA:
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": STRICT_CONTEXT_PROMPT},
    )

def generate_answer(question: str, rag_chain: RetrievalQA) -> str:
    docs = rag_chain.retriever.get_relevant_documents(question)
    if not docs:
        return (
            "I don't know. "
            "I couldn't find any information about that in the provided documents."
        )
    raw = rag_chain.run(question).strip()
    
    cleaned = re.sub(r'(?:Question:?)+\s*$', '', raw, flags=re.IGNORECASE).strip()
    return cleaned    