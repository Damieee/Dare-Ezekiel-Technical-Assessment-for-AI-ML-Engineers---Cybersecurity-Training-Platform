from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings for environment variable loading"""

    # API Settings
    API_TITLE: str = "Qwen Cybersecurity API"
    API_DESCRIPTION: str = "FastAPI application for fine-tuning and inferencing with Qwen 2.5 on cybersecurity data"
    API_VERSION: str = "0.1.0"
    
    # Model Settings
    DEFAULT_MODEL_ID: str = "Qwen/Qwen1.5-0.5B"  # Default to smaller model for dev/testing
    PRODUCTION_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"  # Full model for production with sufficient resources
    MODEL_CACHE_DIR: str = "./model_cache"
    FINE_TUNED_MODEL_PATH: str = "./qwen-cybersecurity-finetuned"
    
    # Training Settings
    TRAINING_BATCH_SIZE: int = 4
    EVAL_BATCH_SIZE: int = 4
    NUM_TRAIN_EPOCHS: int = 3
    LEARNING_RATE: float = 5e-5
    MAX_SEQ_LENGTH: int = 512
    
    # Inference Settings
    MAX_NEW_TOKENS: int = 200
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    REPETITION_PENALTY: float = 1.2
    
    # Hugging Face Hub Settings
    HF_TOKEN: Optional[str] = None
    HF_USERNAME: Optional[str] = None
    
    # Hardware Settings
    USE_GPU: bool = True
    
    class Config:
        env_file = "../../.env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
