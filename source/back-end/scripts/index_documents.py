import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.pdf_parser import extract_chunks_from_pdf
from app.services.watsonx_client import get_embedding
from app.services.chroma_db import add_document


def index_pdf_chunks(pdf_path: str):
    print("Starting PDF content extraction and chunking.")

    try:
        chunks = extract_chunks_from_pdf(pdf_path, chunk_size=500, overlap=100)
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        return

    print(f"Document successfully split into {len(chunks)} chunks.")
    print("Beginning embedding generation and indexing.")

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            add_document(id=f"chunk_{i}", text=chunk, embedding=embedding)
        except Exception as e:
            print(f"Error indexing chunk {i}: {e}")

    print("Indexing process completed.")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    pdf_path = os.path.abspath(os.path.join(current_dir, "..", "assets", "Unleashing_the_Power_of_AI_with_IBM_watsonxai.pdf"))

    if not os.path.exists(pdf_path):
        print(f"PDF not found at path: {pdf_path}")
    else:
        index_pdf_chunks(pdf_path)
