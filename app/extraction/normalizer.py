"""Normalization of extracted feature values."""
import logging
from typing import Any, Dict, Optional

from app.contracts.property_features import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    ORDINAL_USER_TO_MODEL,
    REQUIRED_FEATURES,
)
from app.core.exceptions import NormalizationException

logger = logging.getLogger(__name__)

VALID_FEATURE_NAMES = set(REQUIRED_FEATURES)

class FeatureNormalizer:
    """Normalize extracted feature values from LLM output."""

    @staticmethod
    def normalize_feature(
        feature_name: str, raw_value: Any
    ) -> Optional[Any]:
        """Normalize a single feature value.
        
        Args:
            feature_name: Name of the feature
            raw_value: Raw value from LLM
            
        Returns:
            Normalized value or None if unable to normalize
            
        Raises:
            NormalizationException: If value is clearly invalid
        """
        if feature_name not in VALID_FEATURE_NAMES:
            raise NormalizationException(
                f"Unknown feature name returned by LLM: {feature_name}"
            )

        if raw_value is None or raw_value == "":
            return None

        # Convert to string for cleaning
        value_str = str(raw_value).strip()

        # Numeric features: try to convert to float/int
        if feature_name in NUMERIC_FEATURES:
            return FeatureNormalizer._normalize_numeric(feature_name, value_str)

        # Ordinal features: map user value to model token
        if feature_name in ORDINAL_FEATURES:
            return FeatureNormalizer._normalize_ordinal(feature_name, value_str)

        # Nominal features (Neighborhood): return as-is
        if feature_name in NOMINAL_FEATURES:
            return value_str
        raise NormalizationException(
            f"Feature name is not mapped to a supported feature group: {feature_name}"
        )

    @staticmethod
    def _normalize_numeric(feature_name: str, value_str: str) -> Optional[int | float]:
        """Normalize numeric value.
        
        Args:
            feature_name: Name of numeric feature
            value_str: String value to convert
            
        Returns:
            Converted numeric value or None
            
        Raises:
            NormalizationException: If value cannot be converted
        """
        try:
            # Try int first
            if "." not in value_str:
                return int(value_str)
            return float(value_str)
        except ValueError:
            logger.warning(
                f"Cannot convert {feature_name}={value_str} to numeric"
            )
            raise NormalizationException(
                f"Invalid numeric value for {feature_name}: {value_str}"
            )

    @staticmethod
    def _normalize_ordinal(feature_name: str, value_str: str) -> Optional[str]:
        """Normalize ordinal value using user->model mapping.
        
        Args:
            feature_name: Name of ordinal feature
            value_str: String value to map
            
        Returns:
            Model token (e.g., "Ex", "Gd") or None if not found
            
        Raises:
            NormalizationException: If value not in mapping
        """
        # Try direct match first
        if value_str in ORDINAL_USER_TO_MODEL:
            model_token = ORDINAL_USER_TO_MODEL[value_str]
            return model_token

        # Try case-insensitive match
        for user_key, model_val in ORDINAL_USER_TO_MODEL.items():
            if user_key.lower() == value_str.lower():
                return model_val

        logger.warning(
            f"Ordinal value not in mapping for {feature_name}: {value_str}"
        )
        raise NormalizationException(
            f"Unknown ordinal value for {feature_name}: {value_str}"
        )

    @staticmethod
    def normalize_all(raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all extracted features.
        
        Args:
            raw_features: Dict of raw feature values
            
        Returns:
            Dict with normalized values
            
        Raises:
            NormalizationException: If any value normalization fails fatally
        """
        normalized = {}
        errors = []

        for feature_name, raw_value in raw_features.items():
            if raw_value is None:
                normalized[feature_name] = None
            else:
                try:
                    normalized[feature_name] = FeatureNormalizer.normalize_feature(
                        feature_name, raw_value
                    )
                except NormalizationException as e:
                    errors.append(str(e))
                    # Don't fail immediately; let the service decide what to do
                    normalized[feature_name] = None

        if errors:
            logger.warning(
                f"Normalization errors occurred: {'; '.join(errors)}"
            )

        return normalized
