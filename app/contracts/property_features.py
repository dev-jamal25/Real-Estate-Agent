"""Shared property features contract.

Defines the single source of truth for:
- Required feature names
- Feature groups (numeric, ordinal, nominal)
- Ordinal mappings (user-facing -> model tokens)
- User-friendly descriptions
"""
from typing import Dict, List, Set

# ============================================================================
# Required Features (12 total) - The exact features needed for ML prediction
# ============================================================================

REQUIRED_FEATURES: List[str] = [
    "OverallQual",      # Overall material and finish quality
    "GrLivArea",        # Above grade (ground) living area (in sq. ft.)
    "TotalBsmtSF",      # Total basement area (in sq. ft.)
    "GarageCars",       # Number of cars the garage can hold
    "Neighborhood",     # Physical locations within Ames city limits
    "ExterQual",        # Exterior material quality
    "YearBuilt",        # Original construction date
    "FullBath",         # Full bathrooms above grade
    "KitchenQual",      # Kitchen quality
    "BsmtQual",         # Basement height and quality
    "TotRmsAbvGrd",     # Total rooms above grade (does not include bathrooms)
    "LotArea",          # Lot size (in square feet)
]

# ============================================================================
# Feature Groups
# ============================================================================

NUMERIC_FEATURES: Set[str] = {
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea",
    "OverallQual",
}

ORDINAL_FEATURES: Set[str] = {
    "ExterQual",
    "KitchenQual",
    "BsmtQual",
}

NOMINAL_FEATURES: Set[str] = {
    "Neighborhood",
}

# Sanity check
assert len(NUMERIC_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_FEATURES) == len(
    REQUIRED_FEATURES
), "Feature group partitions don't cover all required features"

# ============================================================================
# Ordinal Mapping: User-facing values -> Model tokens
# ============================================================================

ORDINAL_USER_TO_MODEL: Dict[str, str] = {
    "None": "None",
    "Poor": "Po",
    "Fair": "Fa",
    "Typical": "TA",
    "Average": "TA",
    "Typical/Average": "TA",
    "Good": "Gd",
    "Excellent": "Ex",
}

# Reverse mapping for validation
ORDINAL_MODEL_TOKENS: Set[str] = set(ORDINAL_USER_TO_MODEL.values())

# ============================================================================
# Feature Descriptions (user-friendly labels)
# ============================================================================

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "OverallQual": "Overall quality (1-10 scale)",
    "GrLivArea": "Living area above ground (in sq. ft.)",
    "TotalBsmtSF": "Total basement area (in sq. ft.)",
    "GarageCars": "Number of garage spaces",
    "Neighborhood": "Neighborhood name",
    "ExterQual": "Exterior quality (Poor/Fair/Typical/Good/Excellent)",
    "YearBuilt": "Year built",
    "FullBath": "Number of full bathrooms",
    "KitchenQual": "Kitchen quality (Poor/Fair/Typical/Good/Excellent)",
    "BsmtQual": "Basement quality (None/Poor/Fair/Typical/Good/Excellent)",
    "TotRmsAbvGrd": "Total rooms above ground",
    "LotArea": "Lot size (in sq. ft.)",
}
