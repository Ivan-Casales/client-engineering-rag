from app.core.config import settings
from app.services.watsonx_client import WatsonXEmbeddings, WatsonXLLM
from app.services.chroma_db import load_vectorstore
from app.services.rag_pipeline import build_rag_chain

# Instantiate WatsonX embeddings
embedding_model = WatsonXEmbeddings()

# Load vector store from persistent directory
vectorstore = load_vectorstore(
    embedding_model=embedding_model,
    persist_directory=settings.CHROMA_PERSIST_DIRECTORY
)

# Instantiate the WatsonX LLM
llm = WatsonXLLM(
    model_id=settings.MODEL_ID,
    temperature=settings.TEMPERATURE,
    max_new_tokens=settings.MAX_NEW_TOKENS
)

# Build the RetrievalQA chain
rag_chain = build_rag_chain(vectorstore, llm)