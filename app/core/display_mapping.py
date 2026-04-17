"""Display mapping for user-friendly labels and values.

This module converts internal model-facing feature names and encoded values
into user-friendly display representations for the UI.
"""
from typing import Any, Dict, Optional

# Feature name mappings (internal -> user-friendly)
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "OverallQual": "Overall quality",
    "GrLivArea": "Above-ground living area",
    "TotalBsmtSF": "Total basement area",
    "GarageCars": "Garage spaces",
    "Neighborhood": "Neighborhood",
    "ExterQual": "Exterior quality",
    "YearBuilt": "Year built",
    "FullBath": "Full bathrooms",
    "KitchenQual": "Kitchen quality",
    "BsmtQual": "Basement quality",
    "TotRmsAbvGrd": "Total rooms above ground",
    "LotArea": "Lot size",
}

# Ordinal value mappings (model token -> user-friendly display)
ORDINAL_DISPLAY_VALUES: Dict[str, str] = {
    "Po": "Poor",
    "Fa": "Fair",
    "TA": "Typical/Average",
    "Gd": "Good",
    "Ex": "Excellent",
    "None": "None",
}

# Unit suffixes for numeric features
FEATURE_UNITS: Dict[str, str] = {
    "GrLivArea": "sq ft",
    "TotalBsmtSF": "sq ft",
    "LotArea": "sq ft",
}


def get_feature_display_name(feature_name: str) -> str:
    """Get the user-friendly display name for a feature.
    
    Args:
        feature_name: Internal feature name
        
    Returns:
        User-friendly display name, or original name if not mapped
    """
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def get_feature_display_value(feature_name: str, value: Any) -> str:
    """Format a feature value for user-friendly display.
    
    Args:
        feature_name: Internal feature name
        value: Raw feature value
        
    Returns:
        Formatted display string
    """
    if value is None:
        return "Not specified"
    
    # Handle ordinal features
    if feature_name in ["ExterQual", "KitchenQual", "BsmtQual"]:
        return ORDINAL_DISPLAY_VALUES.get(str(value), str(value))
    
    # Handle overall quality with scale
    if feature_name == "OverallQual":
        try:
            qual_int = int(value)
            return f"{qual_int}/10"
        except (ValueError, TypeError):
            return str(value)
    
    # Handle numeric features with units
    if feature_name in FEATURE_UNITS:
        try:
            num_val = float(value)
            # Format with comma separator for thousands
            formatted = f"{num_val:,.0f}"
            unit = FEATURE_UNITS[feature_name]
            return f"{formatted} {unit}"
        except (ValueError, TypeError):
            return str(value)
    
    # Handle year
    if feature_name == "YearBuilt":
        try:
            return str(int(value))
        except (ValueError, TypeError):
            return str(value)
    
    # Handle counts (bathrooms, rooms, garage spaces)
    if feature_name in ["FullBath", "TotRmsAbvGrd", "GarageCars"]:
        try:
            return str(int(value))
        except (ValueError, TypeError):
            return str(value)
    
    # Default: return as string
    return str(value)


def format_feature_summary(features: Dict[str, Any]) -> str:
    """Format a dict of features as a user-friendly summary.
    
    Args:
        features: Dict of feature_name -> value
        
    Returns:
        Formatted summary string
    """
    if not features:
        return "(No features captured yet)"
    
    lines = []
    for feature_name in sorted(features.keys()):
        value = features[feature_name]
        display_name = get_feature_display_name(feature_name)
        display_value = get_feature_display_value(feature_name, value)
        lines.append(f"**{display_name}:** {display_value}")
    
    return "\n".join(lines)


def get_fallback_reply() -> str:
    """Get a safe fallback reply when the backend reply is empty or missing.
    
    Returns:
        Fallback message for user
    """
    return (
        "I've captured some details about the property. "
        "Please tell me more so I can provide an accurate estimate."
    )
