from docling import SimpleDocument
from typing import List

def extract_chunks_from_pdf(path: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    doc = SimpleDocument.from_pdf(path)
    full_text = doc.raw_content

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
