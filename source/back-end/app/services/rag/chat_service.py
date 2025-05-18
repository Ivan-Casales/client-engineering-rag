from typing import List, Dict, Tuple
import re
from app.services.container import llm, vectorstore, reranker
from app.services.utility.prompt_chat import STRICT_CONTEXT_PROMPT_CHAT
from app.services.utility.security import sanitize_input

def _format_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not history:
        return ""

    recent = history[-max_turns:]
    blocks = []
    for turn in recent:
        blocks.append(f"User: {turn['user']}")
        blocks.append(f"Assistant: {turn['assistant']}")
    return "\n".join(blocks)


def _clean_output(raw: str) -> str:
    raw = re.split(r"\n(?:User|Assistant):", raw, maxsplit=1)[0]
    cleaned = re.sub(
        r"\b(?:Question|Answer|Final answer)\s*:\s*$", "", raw.strip(), flags=re.IGNORECASE
    )
    return cleaned.strip()

def process_chat(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    try:
        history = history or []

        message = sanitize_input(message)

        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(message)

        if not docs:
            answer = (
                "I don't know. I couldn't find any information about that in the provided documents."
            )
        else:
            docs = reranker.rerank_documents(message, docs, 5)

            docs_block = "\n\n".join(doc.page_content for doc in docs)
            dialogue_block = _format_history(history, 5)

            context_block = f"{dialogue_block}\n\n{docs_block}" if dialogue_block else docs_block

            prompt = STRICT_CONTEXT_PROMPT_CHAT.format(context=context_block, question=message.strip())

            raw_answer = llm(
                prompt,
                stop=["\nUser:", "\nAssistant:", "\Question:", "\Answer:"]
            )

            answer = _clean_output(raw_answer)

        updated_history = history + [{"user": message, "assistant": answer}]
        return updated_history, answer

    except Exception as e:
        raise RuntimeError(f"Chat processing failed: {e}")

