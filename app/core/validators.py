"""Feature validation utilities."""
from typing import Any, Dict, List, Optional, Set

from app.core.constants import (
    FEATURE_GUIDANCE,
    FLOAT_FEATURES,
    INTEGER_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    ORDINAL_MODEL_TO_USER,
    ORDINAL_USER_TO_MODEL,
    REQUIRED_FEATURES,
)


def is_valid_feature_name(name: str) -> bool:
    """Check if a feature name is in the required features list."""
    return name in REQUIRED_FEATURES


def get_supported_feature_names() -> List[str]:
    """Return the list of required feature names."""
    return REQUIRED_FEATURES.copy()


def is_ordinal_feature(name: str) -> bool:
    """Check if a feature is ordinal (quality-based)."""
    return name in ORDINAL_FEATURES


def is_numeric_feature(name: str) -> bool:
    """Check if a feature is numeric."""
    return name in NUMERIC_FEATURES


def is_nominal_feature(name: str) -> bool:
    """Check if a feature is nominal (categorical)."""
    return name in NOMINAL_FEATURES


def map_user_quality_to_model(user_value: str) -> Optional[str]:
    """
    Map user-facing quality label to model-facing token.
    
    Returns None if the user_value is not recognized.
    """
    return ORDINAL_USER_TO_MODEL.get(user_value)


def map_model_quality_to_user(model_value: str) -> Optional[str]:
    """
    Map model-facing token back to user-facing label.
    
    Returns None if the model_value is not recognized.
    """
    return ORDINAL_MODEL_TO_USER.get(model_value)


def normalize_numeric_value(value: Any, feature_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely coerce a value to numeric form.
    
    Returns None if the value cannot be coerced.
    
    For integer fields (OverallQual, GarageCars, YearBuilt, FullBath, TotRmsAbvGrd),
    returns int if possible. For float fields (GrLivArea, TotalBsmtSF, LotArea), returns float.
    
    Args:
        value: The value to normalize
        feature_name: Optional feature name to determine if int or float. If None, returns float.
    
    Returns:
        Normalized numeric value (int or float), or None if invalid
    """
    if value is None:
        return None
    
    # Determine if this should be an int or float
    is_integer_field = feature_name in INTEGER_FEATURES if feature_name else False
    
    # Try to coerce
    try:
        if isinstance(value, bool):
            # Reject booleans
            return None
        elif isinstance(value, int):
            return int(value) if is_integer_field else float(value)
        elif isinstance(value, float):
            # Check if it's a clean integer if it's an integer field
            if is_integer_field:
                if value == int(value):
                    return int(value)
                else:
                    # Fractional value for an integer field is invalid
                    return None
            return value
        elif isinstance(value, str):
            stripped = value.strip()
            if is_integer_field:
                # Try to parse as int
                as_float = float(stripped)
                if as_float == int(as_float):
                    return int(as_float)
                else:
                    return None
            else:
                return float(stripped)
    except (ValueError, TypeError):
        return None
    
    return None


def normalize_ordinal_value(value: Any) -> Optional[str]:
    """
    Normalize and validate an ordinal quality value.
    
    Accepts user-facing labels and maps them to model tokens.
    Returns None if the value is not recognized.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    
    # Try to map from user-facing labels
    mapped = map_user_quality_to_model(value)
    if mapped is not None:
        return mapped
    
    # If already a model token, validate and return
    if value in ORDINAL_MODEL_TO_USER:
        return value
    
    return None


def normalize_nominal_value(value: Any) -> Optional[str]:
    """
    Normalize a nominal (categorical) value.
    
    For now, just ensure it's a non-empty string.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def normalize_feature_value(feature_name: str, value: Any) -> Optional[Any]:
    """
    Normalize a feature value based on its type.
    
    Returns the normalized value or None if invalid/missing.
    """
    if not is_valid_feature_name(feature_name):
        return None
    
    if is_numeric_feature(feature_name):
        return normalize_numeric_value(value, feature_name=feature_name)
    elif is_ordinal_feature(feature_name):
        return normalize_ordinal_value(value)
    elif is_nominal_feature(feature_name):
        return normalize_nominal_value(value)
    
    return None


def compute_missing_features(
    extracted_features: Dict[str, Any],
    current_known_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Compute which required features are still missing.
    
    Args:
        extracted_features: Newly extracted features from this turn
        current_known_state: Previously known feature values (from earlier turns or overrides)
    
    Returns:
        List of feature names that are still missing
    """
    if current_known_state is None:
        current_known_state = {}
    
    # Combine current state and newly extracted features
    combined = {**current_known_state, **extracted_features}
    
    # Find features with no value
    missing = []
    for feature_name in REQUIRED_FEATURES:
        if feature_name not in combined or combined[feature_name] is None:
            missing.append(feature_name)
    
    return missing


def is_complete(
    extracted_features: Dict[str, Any],
    current_known_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Check if all required features are present.
    
    Args:
        extracted_features: Newly extracted features
        current_known_state: Previously known feature values
    
    Returns:
        True if all required features have non-None values
    """
    missing = compute_missing_features(extracted_features, current_known_state)
    return len(missing) == 0


def pick_follow_up_features(missing_features: List[str], count: int = 3) -> List[str]:
    """
    Select high-priority missing features to ask the user about.
    
    Prioritizes numeric features that are easier to answer, and ordinal features next.
    
    Args:
        missing_features: List of missing feature names
        count: Number of features to return (default 3)
    
    Returns:
        Up to `count` highest-priority missing features
    """
    if not missing_features:
        return []
    
    # Sort: numeric first (easier for user), then ordinal, then nominal
    priority_order = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    sorted_missing = sorted(
        missing_features,
        key=lambda f: priority_order.index(f) if f in priority_order else 999,
    )
    
    return sorted_missing[:count]


def build_follow_up_message(missing_features: List[str]) -> str:
    """
    Build a user-friendly follow-up message asking for the most important missing features.
    
    Args:
        missing_features: List of missing feature names
    
    Returns:
        A conversational follow-up message
    """
    if not missing_features:
        return "All required features collected. Ready to predict."
    
    # Pick top 3-4 high-priority features
    to_ask = pick_follow_up_features(missing_features, count=4)
    
    # Build message using FEATURE_GUIDANCE
    parts = [FEATURE_GUIDANCE.get(f, f) for f in to_ask]
    
    if len(parts) == 1:
        msg = f"I still need the {parts[0]}."
    elif len(parts) == 2:
        msg = f"I still need the {parts[0]} and the {parts[1]}."
    else:
        msg = f"I still need the {', '.join(parts[:-1])}, and the {parts[-1]}."
    
    return msg

