from typing import List
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain_chroma import Chroma

def build_vectorstore(
    documents: List[Document],
    embedding_model: Embeddings,
    persist_directory: str
) -> Chroma:
    return Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_directory
    )

def load_vectorstore(
    embedding_model: Embeddings,
    persist_directory: str
) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
