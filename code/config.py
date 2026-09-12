import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    novita_api_key: str = ""
    novita_base_url: str = "https://api.novita.ai/openai"
    model_name: str = "zai-org/glm-5.3-flash"
    
    base_dir: Path = Path(__file__).resolve().parent.parent
    dataset_dir: Path = base_dir / "dataset"
    cache_dir: Path = base_dir / "code" / ".cache"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

cfg = Settings()

os.makedirs(cfg.cache_dir, exist_ok=True)
