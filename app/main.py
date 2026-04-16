"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import dependencies, routes
from app.config import settings
from app.llm.client import GeminiClient
from app.llm.extractor import Stage1Extractor
from app.ml.loader import ArtifactLoader, ArtifactLoadError
from app.ml.predictor import Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    try:
        # Load ML artifact
        logger.info(f"Loading artifact from {settings.model_path}")
        artifact = ArtifactLoader.load(settings.model_path)
        dependencies.set_artifact(artifact)
        logger.info("Artifact loaded successfully")

        # Initialize predictor
        predictor = Predictor(artifact)
        dependencies.set_predictor(predictor)
        logger.info("Predictor initialized")

        # Initialize LLM client and extractor
        # Use mock or real LLM based on configuration
        logger.info(f"Initializing LLM client (use_mock={settings.llm_use_mock})")
        llm_client = GeminiClient(
            use_mock=settings.llm_use_mock,
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
            enable_fallback=settings.llm_enable_fallback,
            fallback_on_quota=settings.llm_fallback_on_quota,
            fallback_on_timeout=settings.llm_fallback_on_timeout,
            fallback_on_5xx=settings.llm_fallback_on_5xx,
            ollama_enabled=settings.ollama_enabled,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            ollama_timeout=settings.ollama_timeout,
        )
        extractor = Stage1Extractor(llm_client=llm_client, prompt_version="v1")
        dependencies.set_extractor(extractor)
        mode_str = "mock" if settings.llm_use_mock else "real Gemini"
        if settings.llm_enable_fallback and settings.ollama_enabled:
            mode_str += f" (with Ollama fallback at {settings.ollama_base_url})"
        logger.info(f"Extractor initialized with {mode_str} LLM client")

        logger.info("Application startup complete")
    except ArtifactLoadError as e:
        logger.error(f"Failed to load artifact during startup: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during startup: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AI Real Estate Agent",
        description="LLM + ML prediction pipeline for property valuation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(routes.router, tags=["predictions"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

