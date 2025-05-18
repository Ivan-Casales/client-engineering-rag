from langchain.prompts import PromptTemplate

TEST_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
Context:
{context}

Question: {question}

Answer:"""
)
