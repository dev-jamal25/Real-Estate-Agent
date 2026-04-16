"""Request models for API endpoints."""
from typing import Optional

from pydantic import BaseModel, Field

from app.models.feature_models import PropertyFeaturesPartial


class PredictRequest(BaseModel):
    """POST /predict request schema."""

    query: str = Field(
        ...,
        description="Natural language description of the property",
        min_length=1,
        max_length=2000,
    )
    feature_overrides: Optional[PropertyFeaturesPartial] = Field(
        default=None,
        description="Previously known or user-corrected feature values",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "2000 square feet, 2 bathrooms, good kitchen",
                "feature_overrides": {
                    "YearBuilt": 2005,
                    "FullBath": 2,
                },
            }
        }

