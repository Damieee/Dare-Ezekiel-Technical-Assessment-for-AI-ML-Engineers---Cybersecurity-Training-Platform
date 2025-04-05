import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Cybersecurity LLM API"
    
    # Model settings
    BASE_MODEL_ID: str = "Qwen/Qwen2.5-0.5B"
    FINETUNED_MODEL_ID: str = os.getenv("FINETUNED_MODEL_ID", "Qwen/Qwen2.5-0.5B-finetuned-cybersec")
    
    # HuggingFace settings
    HF_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # API settings
    MAX_REQUEST_TOKEN_LENGTH: int = 1024
    MAX_RESPONSE_TOKEN_LENGTH: int = 2048
    DEFAULT_MODEL_TEMPERATURE: float = 0.7

    class Config:
        case_sensitive = True

settings = Settings()
