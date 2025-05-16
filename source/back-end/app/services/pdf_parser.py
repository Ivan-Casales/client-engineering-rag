from docling import SimpleDocument
from typing import List

def extract_chunks_from_pdf(path: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    try:
        doc = SimpleDocument.from_pdf(path)
    except Exception as e:
        raise ValueError(f"Error leyendo PDF en {path}: {e}")
    full_text = doc.raw_content

    paragraphs = full_text.split("\n\n")
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + para + "\n\n"
            else:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start += chunk_size - overlap
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks