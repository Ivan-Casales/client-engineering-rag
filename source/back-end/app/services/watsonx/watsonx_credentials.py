from ibm_watsonx_ai import Credentials
from app.core.config import settings

watsonx_credentials = Credentials(
    url=settings.WATSONX_URL,
    api_key=settings.WATSONX_APIKEY
)