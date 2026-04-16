"""Schemas for prediction service.

Defines the complete validated feature payload for model inference.
This is separate from Stage 1 extraction schemas which handle partial features.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class CompletePropertyFeatures(BaseModel):
    """Complete property features validated for prediction.
    
    All 12 required fields must be present and model-ready.
    This schema is used ONLY after Stage 1 extraction is complete.
    """

    OverallQual: int = Field(
        ..., ge=1, le=10, description="Overall quality (1-10)"
    )
    GrLivArea: int = Field(
        ..., gt=0, description="Above grade living area (sq ft)"
    )
    TotalBsmtSF: int = Field(
        ..., ge=0, description="Total basement area (sq ft)"
    )
    GarageCars: int = Field(
        ..., ge=0, description="Garage car capacity"
    )
    Neighborhood: str = Field(
        ..., min_length=1, description="Neighborhood name"
    )
    ExterQual: str = Field(
        ..., min_length=1, description="Exterior quality (Po/Fa/TA/Gd/Ex)"
    )
    YearBuilt: int = Field(
        ..., ge=1800, le=2100, description="Year built"
    )
    FullBath: int = Field(
        ..., ge=0, description="Full bathrooms above grade"
    )
    KitchenQual: str = Field(
        ..., min_length=1, description="Kitchen quality (Po/Fa/TA/Gd/Ex)"
    )
    BsmtQual: str = Field(
        ..., min_length=1, description="Basement quality (None/Po/Fa/TA/Gd/Ex)"
    )
    TotRmsAbvGrd: int = Field(
        ..., ge=1, description="Total rooms above ground"
    )
    LotArea: int = Field(
        ..., gt=0, description="Lot size (sq ft)"
    )

    @field_validator("ExterQual", "KitchenQual", "BsmtQual")
    @classmethod
    def validate_ordinal_tokens(cls, v: str) -> str:
        """Ensure ordinal fields are model tokens (Po/Fa/TA/Gd/Ex/None)."""
        valid_tokens = {"None", "Po", "Fa", "TA", "Gd", "Ex"}
        if v not in valid_tokens:
            raise ValueError(f"Must be one of {valid_tokens}, got {v}")
        return v


class PredictionArtifact(BaseModel):
    """Wrapper for loaded ML artifact.
    
    Contains model pipeline and metadata.
    """

    model: Any = Field(
        ..., description="Fitted sklearn pipeline or model object"
    )
    features: list = Field(
        ..., description="Ordered list of 12 feature names for model"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Model training metadata"
    )
    global_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Global statistics for interpretation"
    )


class PredictionOutput(BaseModel):
    """Prediction service output.
    
    Contains the predicted price and supporting statistics.
    """

    predicted_price: float = Field(
        ..., ge=0, description="Predicted sale price"
    )
    global_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Global statistics (mean, std, etc)"
    )
