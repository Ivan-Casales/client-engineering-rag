from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List

def extract_chunks_from_pdf(path: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Load a PDF file and split it into text chunks using LangChain utilities.

    Parameters:
    - path (str): Path to the PDF file.
    - chunk_size (int): Maximum number of characters per chunk (default is 500).
    - overlap (int): Number of overlapping characters between chunks (default is 100).

    Returns:
    - List[str]: A list of text chunks extracted from the PDF.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_docs = splitter.split_documents(documents)

    return [doc.page_content for doc in split_docs]
