import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    SADTALKER_API_KEY: str = os.getenv("SADTALKER_API_KEY", "")
    
    # API URLs
    SADTALKER_API_URL: str = os.getenv(
        "SADTALKER_API_URL", 
        "https://api.sadtalker.io/generate"
    )
    
    # Application settings
    APP_NAME: str = "AI Avatar Speech & Animation"
    APP_VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    STATIC_DIR: Path = BASE_DIR / "static"
    DEFAULT_AVATAR: str = "default_avatar.jpg"
    
    # Audio settings
    MAX_AUDIO_SIZE_MB: int = 25  # Maximum audio file size in MB
    SUPPORTED_AUDIO_FORMATS: list = ["mp3", "wav", "m4a", "ogg"]
    
    # Video settings
    VIDEO_WIDTH: int = 512
    VIDEO_HEIGHT: int = 512
    VIDEO_FPS: int = 25
    
    # Performance settings
    TIMEOUT_SECONDS: int = 300  # Timeout for API requests
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create required directories
settings.OUTPUT_DIR.mkdir(exist_ok=True)
settings.STATIC_DIR.mkdir(exist_ok=True)
