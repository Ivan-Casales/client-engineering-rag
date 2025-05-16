from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    WATSONX_URL: str
    WATSONX_PROJECT_ID: str
    WATSONX_APIKEY: str
    CHROMA_PERSIST_DIRECTORY: str = ".chromadb"

    @field_validator("WATSONX_URL")
    def validate_url(cls, v):
        if not v.startswith("https://"):
            raise ValueError("WATSONX_URL must start with 'https://'")
        return v

    @field_validator("WATSONX_APIKEY")
    def validate_apikey(cls, v):
        if len(v) < 10:
            raise ValueError("WATSONX_APIKEY seems too short")
        return v

    @field_validator("CHROMA_PERSIST_DIRECTORY")
    def validate_chroma_dir(cls, v):
        if not v:
            raise ValueError("CHROMA_PERSIST_DIRECTORY must be provided")
        return v

    class Config:
        env_file = ".env"

settings = Settings()
