from typing import List, Dict, Tuple
from langchain.chains import ConversationalRetrievalChain
from app.services.container import llm, vectorstore

def process_chat(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    try:
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )

        result = conv_chain({
            "question": message,
            "chat_history": [(turn["user"], turn["assistant"]) for turn in history]
        })

        updated_history = history + [{"user": message, "assistant": result["answer"]}]

        return updated_history, result["answer"]

    except Exception as e:
        raise RuntimeError(f"Chat processing failed: {e}")

