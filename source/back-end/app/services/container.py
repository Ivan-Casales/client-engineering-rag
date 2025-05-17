from app.core.config import settings
from app.services.watsonx_client import WatsonXEmbeddings, WatsonXLLM
from app.services.chroma_db import load_vectorstore
from app.services.reranker import ReRanker
from .utility.prompt_templates import STRICT_CONTEXT_PROMPT
from langchain.chains import RetrievalQA

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": STRICT_CONTEXT_PROMPT},
    )

# Instantiate the ReRanker
reranker = ReRanker()