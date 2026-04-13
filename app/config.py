"""Application configuration from environment variables."""
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    # FastAPI Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS Origins
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # LLM Configuration
    llm_api_key: str = ""  # TODO: Load from .env
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"

    # ML Model Paths
    model_path: str = "artifacts/best_model.pkl"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    stats_path: str = "artifacts/stats.json"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Prompts
    prompt_extract_v1_path: str = "prompts/extract_v1.txt"
    prompt_extract_v2_path: str = "prompts/extract_v2.txt"
    prompt_interpret_v1_path: str = "prompts/interpret_v1.txt"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
