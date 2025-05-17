from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.llms.base import LLM

def build_rag_chain(vectorstore: Chroma, llm: LLM) -> RetrievalQA:
    """
    Construct a RetrievalQA chain using a vector store and an LLM.

    Parameters:
    - vectorstore (Chroma): The Chroma vector database instance.
    - llm (LLM): A LangChain-compatible language model.

    Returns:
    - RetrievalQA: A LangChain RetrievalQA chain object.
    """
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def generate_answer(question: str, rag_chain: RetrievalQA) -> str:
    result = rag_chain.run(question)
    return result