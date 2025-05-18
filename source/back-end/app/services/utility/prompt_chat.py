from langchain.prompts import PromptTemplate

STRICT_CONTEXT_PROMPT_CHAT = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
You are an AI assistant. Use **only** the context below to answer the question.
If the answer is not contained in the context, reply **exactly**:
"I don't know. I couldn't find any information about that in the provided documents."
Output rules (VERY IMPORTANT): Answer **only** the final user question,
do **not** invent additional questions or answers,
do **not** include the tags "User:" or "Assistant:" in your reply,
if unsure, use the fallback phrase above verbatim.

Context:
{context}

Question: {question}

Answer:"""
)
