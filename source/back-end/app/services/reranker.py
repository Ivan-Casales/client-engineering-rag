from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.cross_encoder = CrossEncoder(model_name)

    def rerank_documents(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        ranked_docs = [
            doc for _, doc in sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        return ranked_docs[:top_k]
