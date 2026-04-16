"""Feature extraction data models."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PropertyFeaturesPartial(BaseModel):
    """
    Partial property features model.
    
    All fields are optional to represent either:
    - Newly extracted features (only filled fields from this turn)
    - Accumulated state (cumulative known values)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "OverallQual": 8,
                "GrLivArea": 2000.0,
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Gd",
            }
        }
    )

    OverallQual: Optional[int] = Field(
        default=None,
        description="Overall material and finish quality (1-10)",
    )
    GrLivArea: Optional[float] = Field(
        default=None,
        description="Above-ground living area in square feet",
    )
    TotalBsmtSF: Optional[float] = Field(
        default=None,
        description="Total basement area in square feet",
    )
    GarageCars: Optional[int] = Field(
        default=None,
        description="Number of cars the garage fits",
    )
    Neighborhood: Optional[str] = Field(
        default=None,
        description="Ames neighborhood name",
    )
    ExterQual: Optional[str] = Field(
        default=None,
        description="Exterior quality (None, Po, Fa, TA, Gd, Ex)",
    )
    YearBuilt: Optional[int] = Field(
        default=None,
        description="Year the house was built",
    )
    FullBath: Optional[int] = Field(
        default=None,
        description="Number of full bathrooms",
    )
    KitchenQual: Optional[str] = Field(
        default=None,
        description="Kitchen quality (None, Po, Fa, TA, Gd, Ex)",
    )
    BsmtQual: Optional[str] = Field(
        default=None,
        description="Basement quality (None, Po, Fa, TA, Gd, Ex)",
    )
    TotRmsAbvGrd: Optional[int] = Field(
        default=None,
        description="Total rooms above grade",
    )
    LotArea: Optional[float] = Field(
        default=None,
        description="Lot size in square feet",
    )


class PropertyFeaturesComplete(BaseModel):
    """
    Complete property features model.
    
    All 12 required fields are mandatory (no Optional).
    Used for final validation before prediction.
    Ensures strict completeness: no nulls allowed.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "OverallQual": 8,
                "GrLivArea": 2000.0,
                "TotalBsmtSF": 1000.0,
                "GarageCars": 2,
                "Neighborhood": "CollgCr",
                "ExterQual": "Gd",
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Gd",
                "BsmtQual": "Gd",
                "TotRmsAbvGrd": 8,
                "LotArea": 8200.0,
            }
        }
    )

    OverallQual: int = Field(
        description="Overall material and finish quality (1-10)",
    )
    GrLivArea: float = Field(
        description="Above-ground living area in square feet",
    )
    TotalBsmtSF: float = Field(
        description="Total basement area in square feet",
    )
    GarageCars: int = Field(
        description="Number of cars the garage fits",
    )
    Neighborhood: str = Field(
        description="Ames neighborhood name",
    )
    ExterQual: str = Field(
        description="Exterior quality (None, Po, Fa, TA, Gd, Ex)",
    )
    YearBuilt: int = Field(
        description="Year the house was built",
    )
    FullBath: int = Field(
        description="Number of full bathrooms",
    )
    KitchenQual: str = Field(
        description="Kitchen quality (None, Po, Fa, TA, Gd, Ex)",
    )
    BsmtQual: str = Field(
        description="Basement quality (None, Po, Fa, TA, Gd, Ex)",
    )
    TotRmsAbvGrd: int = Field(
        description="Total rooms above grade",
    )
    LotArea: float = Field(
        description="Lot size in square feet",
    )


