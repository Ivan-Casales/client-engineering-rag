import sys

# To avoid circular import issues, we need to remove the modules from sys.modules
# before importing them.
for module in (
    "app.core.config",
    "app.services.watsonx_client",
    "app.services.chroma_db",
    "app.services.rag_pipeline",
):
    sys.modules.pop(module, None)

from app.services.watsonx_client import WatsonXEmbeddings, WatsonXLLM

def get_embedding(text: str) -> list[float]:
    return WatsonXEmbeddings().embed_query(text)

def generate_answer_with_context(context: str, question: str) -> str:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return WatsonXLLM()(prompt)

def rerank_documents(question: str, documents: list[str]) -> list[str]:
    emb = WatsonXEmbeddings()
    qv = emb.embed_query(question)
    dvs = emb.embed_documents(documents)
    sims = [
        sum(a*b for a,b in zip(qv, dv)) / ((sum(a*a for a in qv)**0.5)*(sum(b*b for b in dv)**0.5) or 1)
        for dv in dvs
    ]
    return [doc for _,doc in sorted(zip(sims, documents), reverse=True)]
