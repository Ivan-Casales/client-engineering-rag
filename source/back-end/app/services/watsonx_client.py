from typing import List, Optional
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings as WatsonxEmbedClient, ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from app.core.config import settings

# Initialize IBM Watsonx credentials
credentials = Credentials(
    url=settings.WATSONX_URL,
    api_key=settings.WATSONX_APIKEY
)

class WatsonXEmbeddings(Embeddings):
    def __init__(self):
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {'input_text': False}
        }
        self.client = WatsonxEmbedClient(
            model_id=settings.EMBEDDING_MODEL_ID,
            params=embed_params,
            credentials=credentials,
            project_id=settings.WATSONX_PROJECT_ID,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed_documents(texts=texts)
        return [r.get("embedding") if isinstance(r, dict) else r for r in response]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class WatsonXLLM(LLM):
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
        # Determine parameters, falling back to settings
        model_id_val = model_id or settings.MODEL_ID
        temperature_val = temperature if temperature is not None else settings.TEMPERATURE
        max_tokens_val = max_new_tokens if max_new_tokens is not None else settings.MAX_NEW_TOKENS

        # Initialize BaseModel (pydantic) fields
        super().__init__(
            model_id=model_id_val,
            temperature=temperature_val,
            max_new_tokens=max_tokens_val
        )

        # Initialize the WatsonX ModelInference client
        self.client = ModelInference(
            model_id=self.model_id,
            credentials=credentials,
            project_id=settings.WATSONX_PROJECT_ID,
        )

    @property
    def _llm_type(self) -> str:
        return "watsonx"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.generate(
            prompt,
            params={
                "temperature": self.temperature,
                "decoding_method": "greedy",
                "max_new_tokens": self.max_new_tokens
            }
        )
        return response["results"][0]["generated_text"].strip()

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
