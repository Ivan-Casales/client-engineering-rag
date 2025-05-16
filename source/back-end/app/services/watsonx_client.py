from app.core.config import settings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingModels

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
        "task": "classification",
        "temperature": 0
    }
)

qa_model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    credentials=credentials,
    project_id=settings.WATSONX_PROJECT_ID,
    params={
        "decoding_method": "beam_search",
        "num_beams": 4,
        "max_new_tokens": 300,
        "no_repeat_ngram_size": 3,
        "temperature": 0.2
    }
)

def get_embedding(text: str) -> list[float]:
    results = embedding_client.embed_documents(texts=[text])
    first = results[0]
    if isinstance(first, dict) and 'embedding' in first:
        return first['embedding']

    return first

def rerank_documents(question: str, documents: list[str]) -> list[str]:
    prompts = [f"Question: {question}\nContext: {doc}" for doc in documents]
    results = rerank_model.generate(prompts)
    scored = list(zip(documents, results["results"]))
    scored.sort(key=lambda x: x[1]["prediction"], reverse=True)
    return [doc for doc, _ in scored]

def generate_answer_with_context(context: str, question: str) -> str:
    prompt = (
        "Answer the question based on the context.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = qa_model.generate(prompt)
    return response["results"][0]["generated_text"]