from app.services.pdf_parser import extract_chunks_from_pdf
from app.services.chroma_db import load_vectorstore
from app.services.watsonx_client import WatsonXEmbeddings
from app.core.config import settings
from langchain.schema import Document
import tempfile
import os
from typing import Tuple


def process_pdf_upload(file_bytes: bytes) -> Tuple[int, str]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        chunks = extract_chunks_from_pdf(tmp_path)

        # Wrap each chunk into a LangChain Document
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Load the vector store and index the documents
        embedding_model = WatsonXEmbeddings()
        vectorstore = load_vectorstore(
            embedding_model,
            settings.CHROMA_PERSIST_DIRECTORY
        )
        vectorstore.add_documents(documents)

        return len(documents), ""

    except Exception as e:
        return 0, str(e)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)