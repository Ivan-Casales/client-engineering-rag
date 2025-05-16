from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    WATSONX_URL: str
    WATSONX_PROJECT_ID: str
    WATSONX_APIKEY: str

    class Config:
        env_file = ".env"

settings = Settings()
