"""Response models for API endpoints."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractedFeatures(BaseModel):
    """Features extracted by Stage 1 LLM."""

    features: Dict[str, Any] = Field(
        description="Extracted feature values from user query"
    )
    missing_fields: List[str] = Field(
        description="Required fields not confidently extracted"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0 to 1.0)",
    )


class PredictionResult(BaseModel):
    """ML prediction output."""

    predicted_price: float = Field(description="Predicted house price in USD")
    model_name: str = Field(description="ML model type used for prediction")
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Prediction confidence bounds (lower, upper)",
    )


class Interpretation(BaseModel):
    """Prediction interpretation by Stage 2 LLM."""

    narrative: str = Field(
        description="Natural language explanation of the prediction"
    )
    key_drivers: List[str] = Field(
        description="Main features driving the price prediction"
    )
    market_context: Optional[str] = Field(
        default=None,
        description="How prediction compares to market statistics",
    )


class PredictResponse(BaseModel):
    """Complete /predict endpoint response."""

    query: str = Field(description="Original user query")
    extraction: ExtractedFeatures = Field(description="Stage 1 LLM extraction")
    status: str = Field(
        description="Response status: 'incomplete' or 'complete'"
    )
    prediction: Optional[PredictionResult] = Field(
        default=None,
        description="ML prediction (only if extraction complete)",
    )
    interpretation: Optional[Interpretation] = Field(
        default=None,
        description="Stage 2 LLM interpretation (only if prediction complete)",
    )


class HealthResponse(BaseModel):
    """GET /health endpoint response."""

    status: str = Field(description="Health status")
    message: Optional[str] = Field(default=None)
