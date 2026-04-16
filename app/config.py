"""Application configuration from environment variables.

For this Stage 1 extraction-only slice:
- OpenRouter is the only provider
- No Gemini, no fallback, no multi-provider logic
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore old config vars from previous slices
    )

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # OpenRouter Configuration (Stage 1 extraction)
    openrouter_api_key: str = ""  # Must be set from .env
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/elephant-alpha"

    # Prompts
    prompt_extract_v1_path: str = "prompts/extract_v1.txt"
    prompt_extract_v2_path: str = "prompts/extract_v2.txt"
    prompt_interpret_v1_path: str = "prompts/interpret_v1.txt"

    # ML Artifacts
    artifacts_model_path: str = "artifacts/model.joblib"

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = ["*"]  # Allow all origins for now


settings = Settings()
