"""API request and response schemas for Stage 1 extraction and Stage 2 prediction."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request for Stage 1 extraction prediction.
    
    Attributes:
        query: Latest user message containing property details
        accumulated_features: Known features from previous turns (dict of feature_name -> value)
    """

    query: str = Field(
        ..., min_length=1, description="Latest user message with property details"
    )
    accumulated_features: Dict[str, Any] = Field(
        default_factory=dict, description="Known features from previous turns"
    )


class PredictResponse(BaseModel):
    """Response from Stage 1 extraction and Stage 2 prediction.
    
    If incomplete (is_complete=False): returns extraction response only.
    If complete (is_complete=True): also returns predicted price, global stats, and interpretation.
    
    Attributes:
        accumulated_features: Merged state (previous accumulated + latest extracted)
        extracted_features: Only newly extracted features from this turn
        missing_features: List of feature names still needed
        is_complete: True if all 12 required features are present
        reply: LLM-generated conversational response from Stage 1
        predicted_price: Predicted sale price (only when is_complete=True)
        global_stats: Global statistics (only when is_complete=True)
        interpretation: LLM-generated interpretation (only when is_complete=True)
    """

    accumulated_features: Dict[str, Any] = Field(
        ..., description="Merged state: previous accumulated + latest extracted"
    )
    extracted_features: Dict[str, Any] = Field(
        default_factory=dict, description="Only newly extracted features from this turn"
    )
    missing_features: List[str] = Field(
        ..., description="Feature names still missing"
    )
    is_complete: bool = Field(
        ..., description="True if all 12 required features are present"
    )
    reply: str = Field(
        ..., description="LLM-generated conversational response"
    )
    predicted_price: Optional[float] = Field(
        default=None, description="Predicted sale price (only when is_complete=True)"
    )
    global_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Global statistics (only when is_complete=True)"
    )
    interpretation: Optional[str] = Field(
        default=None, description="LLM-generated interpretation (only when is_complete=True)"
    )


class HealthResponse(BaseModel):
    """Response from health check endpoint.
    
    Attributes:
        status: Always 'healthy' for basic health check
    """

    status: str = Field(default="healthy", description="Health status")
