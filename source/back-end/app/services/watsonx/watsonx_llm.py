from typing import List, Optional
from pydantic import PrivateAttr
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from ibm_watsonx_ai.foundation_models import Embeddings as WatsonxEmbedClient, ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from app.core.config import settings
from .watsonx_credentials import watsonx_credentials

class WatsonXLLM(LLM):
    _default_stop: List[str] = PrivateAttr(default_factory=lambda: ["\nQuestion"])

    model_id: str
    temperature: float
    max_new_tokens: int
    client: Optional[ModelInference] = None

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ):
        model_id_val = model_id or settings.MODEL_ID
        temperature_val = temperature if temperature is not None else settings.TEMPERATURE
        max_tokens_val = max_new_tokens if max_new_tokens is not None else settings.MAX_NEW_TOKENS

        super().__init__(
            model_id=model_id_val,
            temperature=temperature_val,
            max_new_tokens=max_tokens_val
        )

        self.client = ModelInference(
            model_id=self.model_id,
            credentials=watsonx_credentials,
            project_id=settings.WATSONX_PROJECT_ID,
        )

    @property
    def _llm_type(self) -> str:
        return "watsonx"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        params = {
            "temperature": self.temperature,
            "decoding_method": "greedy",
            "max_new_tokens": self.max_new_tokens,
            "stop_sequences": stop if stop is not None else self._default_stop,
        }
        response = self.client.generate(prompt, params=params)
        return response["results"][0]["generated_text"].strip()

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
