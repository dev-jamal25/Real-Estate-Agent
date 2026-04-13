"""Request models for API endpoints."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """POST /predict request schema."""

    query: str = Field(
        ...,
        description="Natural language description of the property",
        min_length=1,
        max_length=2000,
    )
    feature_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-provided feature values (override LLM extraction)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "3-bedroom ranch with garage in a good neighborhood",
                "feature_overrides": {"bedrooms": 3},
            }
        }
