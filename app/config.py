"""Application configuration from environment variables."""
from typing import List

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # FastAPI Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS Origins
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # LLM Configuration (Gemini - Primary)
    gemini_api_key: str = ""  # Optional, for real API usage
    gemini_model_name: str = "gemini-1.5-flash"
    llm_use_mock: bool = False  # Default to mock for testing

    # LLM Fallback Configuration (Ollama - Local Dev Only)
    # These settings enable local Ollama as a fallback provider for Stage 1
    # Intended for development and local testing only, not production
    llm_enable_fallback: bool = True  # Disable fallback by default (opt-in)
    llm_fallback_on_quota: bool = True  # Fallback when Gemini hits quota/rate-limit
    llm_fallback_on_timeout: bool = True  # Fallback when Gemini times out
    llm_fallback_on_5xx: bool = True  # Fallback when Gemini returns 5xx error
    ollama_enabled: bool = False  # Ollama fallback disabled by default
    ollama_base_url: str = "http://localhost:11434"  # Local Ollama endpoint
    ollama_model: str = "llama3.2:1b"  # Lightweight local model
    ollama_timeout: int = 30  # Seconds

    # ML Model Paths
    model_path: str = "artifacts/model.joblib"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    stats_path: str = "artifacts/stats.json"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Prompts
    prompt_extract_v1_path: str = "prompts/extract_v1.txt"
    prompt_extract_v2_path: str = "prompts/extract_v2.txt"
    prompt_interpret_v1_path: str = "prompts/interpret_v1.txt"


settings = Settings()
