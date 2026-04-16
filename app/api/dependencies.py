"""Lightweight dependency injection for API routes."""
from app.config import settings
from app.extraction.service import ExtractionService
from app.interpretation.service import InterpretationService
from app.llm.openrouter_client import OpenRouterClient
from app.prediction.artifact_loader import load_artifact
from app.prediction.service import PredictionService


def get_extraction_service() -> ExtractionService:
    """Provide ExtractionService for API routes.
    
    Returns:
        ExtractionService initialized with OpenRouter client
    """
    client = OpenRouterClient()
    return ExtractionService(client=client)


def get_prediction_service() -> PredictionService:
    """Provide PredictionService for API routes.
    
    Loads artifact once per request.
    
    Returns:
        PredictionService initialized with loaded artifact
        
    Raises:
        FileNotFoundError: If artifact not found
        ValueError: If artifact is invalid
    """
    artifact = load_artifact(settings.artifacts_model_path)
    return PredictionService(artifact=artifact)


def get_interpretation_service() -> InterpretationService:
    """Provide InterpretationService for API routes.
    
    Returns:
        InterpretationService initialized with OpenRouter client
    """
    client = OpenRouterClient()
    return InterpretationService(client=client)
