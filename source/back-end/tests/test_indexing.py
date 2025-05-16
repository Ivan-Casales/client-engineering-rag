import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.pdf_parser import extract_chunks_from_pdf
from app.services.watsonx_client import get_embedding
from app.services.chroma_db import add_document

def test_index_pdf(pdf_path: str):
    print("TEST INDEXING PDF")
    try:
        chunks = extract_chunks_from_pdf(pdf_path)
        print(f"Extracted {len(chunks)} chunks from {pdf_path}")
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return

    success_count = 0
    for idx, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            doc_id = f"{os.path.basename(pdf_path)}-chunk-{idx}"
            add_document(id=doc_id, text=chunk, embedding=embedding)
            success_count += 1
        except Exception as e:
            print(f"  Failed to index chunk {idx}: {e}")

    print(f"Indexed {success_count}/{len(chunks)} chunks successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_indexing.py <path_to_pdf>")
        sys.exit(1)
    test_index_pdf(sys.argv[1])
