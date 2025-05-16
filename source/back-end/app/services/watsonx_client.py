from typing import List
from app.core.config import settings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

credentials = Credentials(
    url=settings.WATSONX_URL,
    api_key=settings.WATSONX_APIKEY
)

embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {
        'input_text': False
    }
}

embedding_client = Embeddings(
    model_id="ibm/slate-30m-english-rtrvr-v2",
    params=embed_params,
    credentials=credentials,
    project_id=settings.WATSONX_PROJECT_ID,
)

rerank_model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    credentials=credentials,
    project_id=settings.WATSONX_PROJECT_ID,
    params={
        "temperature": 0
    }
)

qa_model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    credentials=credentials,
    project_id=settings.WATSONX_PROJECT_ID,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 300
    }
)

def get_embedding(text: str) -> List[float]:
    """
    Compute the embedding vector for the given text.

    Parameters:
    - text (str): The input text to be embedded.

    Returns:
    - List[float]: The embedding vector as a list of floats.
    """
    resp = embedding_client.embed_documents(texts=[text])
    first = resp[0]
    if isinstance(first, dict):
        return first.get("embedding", [])
    return first

def rerank_documents(question: str, documents: list[str]) -> list[str]:
    """
    Rerank a list of documents by relevance to the given question.

    Parameters:
    - question (str): The user question used to evaluate relevance.
    - documents (List[str]): A list of document texts to be reranked.

    Returns:
    - List[str]: Documents sorted by descending relevance.
    """
    prompts = [f"Question: {question}\nContext: {doc}" for doc in documents]
    resp = rerank_model.generate(prompts)

    raw = resp.get("results", resp) if isinstance(resp, dict) else resp

    if (
        not isinstance(raw, list)
        or not raw
        or not isinstance(raw[0], dict)
        or "prediction" not in raw[0]
    ):
        return documents

    scored = list(zip(
        documents,
        [item["prediction"] for item in raw]
    ))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored]

def generate_answer_with_context(context: str, question: str) -> str:
    """
    Generate a conversational answer based on provided context and question.

    Parameters:
    - context (str): Concatenated top documents serving as context.
    - question (str): The user's question to be answered.

    Returns:
    - str: The generated answer text.
    """
    prompt = (
        "System: You are a helpful assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"User: {question}\n"
        "Assistant:"
    )
    resp = qa_model.generate(prompt)
    return resp["results"][0]["generated_text"].strip()