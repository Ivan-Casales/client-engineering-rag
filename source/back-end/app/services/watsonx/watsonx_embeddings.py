from typing import List, Optional
from pydantic import PrivateAttr
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from ibm_watsonx_ai.foundation_models import Embeddings as WatsonxEmbedClient, ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from app.core.config import settings
from .watsonx_credentials import watsonx_credentials

class WatsonXEmbeddings(Embeddings):
    def __init__(self):
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {'input_text': False}
        }
        self.client = WatsonxEmbedClient(
            model_id=settings.EMBEDDING_MODEL_ID,
            params=embed_params,
            credentials=watsonx_credentials,
            project_id=settings.WATSONX_PROJECT_ID,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed_documents(texts=texts)
        return [r.get("embedding") if isinstance(r, dict) else r for r in response]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]