"""API routes: /predict (POST) and /health (GET)."""
from fastapi import APIRouter

from app.models.request_models import PredictRequest
from app.models.response_models import HealthResponse, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    TODO: Verify model and LLM client availability.
    """
    return HealthResponse(status="ok", message="Service running")


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Main prediction endpoint.
    
    Flow:
    1. Run LLM Stage 1 extraction on query
    2. Merge with feature_overrides (overrides win)
    3. Check completeness
    4. If complete: run ML prediction
    5. If complete: run LLM Stage 2 interpretation
    6. Return response with status, extraction, prediction, interpretation
    
    TODO: Implement full prediction pipeline.
    """
    pass
