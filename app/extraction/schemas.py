"""Pydantic schemas for extraction."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class ExtractedFeatures(BaseModel):
    """Partially or fully extracted property features.
    
    All fields are optional because extraction is gradual.
    Values can be raw strings, integers, or floats.
    """

    OverallQual: Optional[Any] = Field(
        None, description="Overall quality"
    )
    GrLivArea: Optional[Any] = Field(
        None, description="Living area (sq. ft.)"
    )
    TotalBsmtSF: Optional[Any] = Field(
        None, description="Basement area (sq. ft.)"
    )
    GarageCars: Optional[Any] = Field(
        None, description="Garage spaces"
    )
    Neighborhood: Optional[Any] = Field(
        None, description="Neighborhood name"
    )
    ExterQual: Optional[Any] = Field(
        None, description="Exterior quality"
    )
    YearBuilt: Optional[Any] = Field(
        None, description="Year built"
    )
    FullBath: Optional[Any] = Field(
        None, description="Number of full bathrooms"
    )
    KitchenQual: Optional[Any] = Field(
        None, description="Kitchen quality"
    )
    BsmtQual: Optional[Any] = Field(
        None, description="Basement quality"
    )
    TotRmsAbvGrd: Optional[Any] = Field(
        None, description="Total rooms above ground"
    )
    LotArea: Optional[Any] = Field(
        None, description="Lot area (sq. ft.)"
    )

    model_config = ConfigDict(extra='forbid')  # forbid additional fields


class Stage1Response(BaseModel):
    """Stage 1 extraction output.
    
    This is the canonical contract between extraction and the next stage.
    - extracted_features: validated and normalized features (some may be None)
    - missing_features: list of feature names still needed
    - is_complete: True if all 12 required features are present
    - reply: LLM-generated conversational response for user
    """
    model_config = ConfigDict(extra='forbid') 
    extracted_features: ExtractedFeatures = Field(
        ..., description="Extracted and normalized features"
    )
    missing_features: List[str] = Field(
        default_factory=list, description="Feature names still missing"
    )
    is_complete: bool = Field(
        ..., description="True if all required features are present"
    )
    reply: str = Field(
        ..., description="LLM-generated conversational reply for the user"
    )


class LLMExtractionOutput(BaseModel):
    """Raw LLM extraction output (before normalization)."""
    
    model_config = ConfigDict(extra="forbid")
    
    extracted_features: Dict[str, Any] = Field(
        default_factory=dict, description="Raw extracted features"
    )
    missing_features: List[str] = Field(
        default_factory=list, description="Features the model identifies as missing"
    )
    reply: str = Field(
        ..., description="Conversational reply from LLM"
    )
