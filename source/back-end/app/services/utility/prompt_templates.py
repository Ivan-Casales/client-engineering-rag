from langchain.prompts import PromptTemplate

STRICT_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
You are an AI assistant. Use **only** the context below to answer the question.
If the answer is not contained in the context, reply **exactly** ""I don't know. I couldn't find any information about that in the provided documents."".

Context:
{context}

Question: {question}

Answer:"""
)
