"""Tests for feature validation."""
import pytest

from app.core.constants import (
    ORDINAL_MODEL_TO_USER,
    ORDINAL_USER_TO_MODEL,
    REQUIRED_FEATURES,
)
from app.core.validators import (
    build_follow_up_message,
    compute_missing_features,
    is_complete,
    is_ordinal_feature,
    is_valid_feature_name,
    map_model_quality_to_user,
    map_user_quality_to_model,
    normalize_feature_value,
    normalize_numeric_value,
    normalize_ordinal_value,
    pick_follow_up_features,
)


class TestFeatureNameValidation:
    """Test feature name validation."""

    def test_valid_feature_names(self):
        """All required features should be valid."""
        for feature in REQUIRED_FEATURES:
            assert is_valid_feature_name(feature)

    def test_invalid_feature_name(self):
        """Non-existent feature names should be invalid."""
        assert not is_valid_feature_name("BedroomCount")
        assert not is_valid_feature_name("OVERALLQUAL")  # case-sensitive
        assert not is_valid_feature_name("RandomField")

    def test_ordinal_feature_check(self):
        """Ordinal features should be correctly identified."""
        assert is_ordinal_feature("ExterQual")
        assert is_ordinal_feature("KitchenQual")
        assert is_ordinal_feature("BsmtQual")
        assert not is_ordinal_feature("YearBuilt")
        assert not is_ordinal_feature("Neighborhood")


class TestOrdinalMapping:
    """Test user-to-model ordinal quality mapping."""

    def test_user_to_model_mapping_all_values(self):
        """All user-facing values should map to correct model tokens."""
        assert map_user_quality_to_model("None") == "None"
        assert map_user_quality_to_model("Poor") == "Po"
        assert map_user_quality_to_model("Fair") == "Fa"
        assert map_user_quality_to_model("Typical/Average") == "TA"
        assert map_user_quality_to_model("Good") == "Gd"
        assert map_user_quality_to_model("Excellent") == "Ex"

    def test_model_to_user_mapping(self):
        """Model tokens should map back to user-facing labels."""
        assert map_model_quality_to_user("None") == "None"
        assert map_model_quality_to_user("Po") == "Poor"
        assert map_model_quality_to_user("Fa") == "Fair"
        assert map_model_quality_to_user("TA") == "Typical/Average"
        assert map_model_quality_to_user("Gd") == "Good"
        assert map_model_quality_to_user("Ex") == "Excellent"

    def test_invalid_ordinal_values(self):
        """Invalid ordinal values should return None."""
        assert map_user_quality_to_model("Mediocre") is None
        assert map_user_quality_to_model("Average") is None
        assert map_user_quality_to_model("") is None
        assert map_model_quality_to_user("XY") is None


class TestNumericNormalization:
    """Test numeric value normalization."""

    def test_normalize_valid_integers(self):
        """Valid integers should normalize correctly."""
        assert normalize_numeric_value(5) == 5.0
        assert normalize_numeric_value(1998) == 1998.0

    def test_normalize_valid_floats(self):
        """Valid floats should normalize correctly."""
        assert normalize_numeric_value(1850.5) == 1850.5
        assert normalize_numeric_value(0.0) == 0.0

    def test_normalize_numeric_strings(self):
        """Numeric strings should be coerced to float."""
        assert normalize_numeric_value("1850") == 1850.0
        assert normalize_numeric_value("2005.5") == 2005.5
        assert normalize_numeric_value("  1850  ") == 1850.0  # with whitespace

    def test_normalize_invalid_numeric(self):
        """Invalid numeric values should return None."""
        assert normalize_numeric_value("abc") is None
        assert normalize_numeric_value("") is None
        assert normalize_numeric_value(None) is None
        assert normalize_numeric_value([1, 2, 3]) is None


class TestOrdinalNormalization:
    """Test ordinal value normalization."""

    def test_normalize_user_facing_ordinals(self):
        """User-facing labels should normalize to model tokens."""
        assert normalize_ordinal_value("Good") == "Gd"
        assert normalize_ordinal_value("Excellent") == "Ex"
        assert normalize_ordinal_value("Fair") == "Fa"

    def test_normalize_model_facing_ordinals(self):
        """Model tokens already normalized should pass through."""
        assert normalize_ordinal_value("Gd") == "Gd"
        assert normalize_ordinal_value("Po") == "Po"
        assert normalize_ordinal_value("None") == "None"

    def test_normalize_invalid_ordinals(self):
        """Invalid ordinal values should return None."""
        assert normalize_ordinal_value("Great") is None
        assert normalize_ordinal_value("") is None
        assert normalize_ordinal_value(None) is None
        assert normalize_ordinal_value(123) is None


class TestFeatureValueNormalization:
    """Test normalization of values based on feature type."""

    def test_normalize_numeric_feature(self):
        """Numeric features should be coerced to float."""
        assert normalize_feature_value("OverallQual", 8) == 8.0
        assert normalize_feature_value("GrLivArea", "1850") == 1850.0
        assert normalize_feature_value("YearBuilt", "2005") == 2005.0

    def test_normalize_ordinal_feature(self):
        """Ordinal features should be mapped from user to model."""
        assert normalize_feature_value("ExterQual", "Good") == "Gd"
        assert normalize_feature_value("KitchenQual", "Excellent") == "Ex"
        assert normalize_feature_value("BsmtQual", "Fair") == "Fa"

    def test_normalize_nominal_feature(self):
        """Nominal features should remain as strings."""
        assert normalize_feature_value("Neighborhood", "NAmes") == "NAmes"
        assert normalize_feature_value("Neighborhood", "  Downtown  ") == "Downtown"

    def test_normalize_invalid_feature_name(self):
        """Unknown feature names should return None."""
        assert normalize_feature_value("UnknownFeature", "value") is None

    def test_normalize_invalid_values_per_type(self):
        """Invalid values for feature type should return None."""
        assert normalize_feature_value("OverallQual", "not_a_number") is None
        assert normalize_feature_value("ExterQual", "Mediocre") is None
        assert normalize_feature_value("Neighborhood", "") is None


class TestComputeMissingFeatures:
    """Test missing feature computation."""

    def test_all_missing_from_scratch(self):
        """With no prior state, all features should be missing."""
        missing = compute_missing_features({})
        assert len(missing) == 12
        assert set(missing) == set(REQUIRED_FEATURES)

    def test_missing_after_partial_extraction(self):
        """After partial extraction, remaining features should be missing."""
        extracted = {
            "YearBuilt": 2005,
            "FullBath": 2,
        }
        missing = compute_missing_features(extracted)
        assert len(missing) == 10
        assert "YearBuilt" not in missing
        assert "FullBath" not in missing

    def test_missing_with_current_state(self):
        """Current known state should reduce missing count."""
        current_state = {
            "YearBuilt": 1998,
            "FullBath": 2,
        }
        extracted = {
            "GarageCars": 2,
        }
        missing = compute_missing_features(extracted, current_state)
        assert len(missing) == 9
        assert "YearBuilt" not in missing
        assert "FullBath" not in missing
        assert "GarageCars" not in missing

    def test_all_present_no_missing(self):
        """With all features present, no features should be missing."""
        all_features = {
            "OverallQual": 7,
            "GrLivArea": 1850.0,
            "TotalBsmtSF": 900.0,
            "GarageCars": 2,
            "Neighborhood": "NAmes",
            "ExterQual": "Gd",
            "YearBuilt": 2005,
            "FullBath": 2,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 7,
            "LotArea": 8200.0,
        }
        missing = compute_missing_features(all_features)
        assert len(missing) == 0


class TestCompletenessCheck:
    """Test completeness checking."""

    def test_incomplete_with_no_features(self):
        """No features should be incomplete."""
        assert not is_complete({})

    def test_incomplete_with_some_features(self):
        """Partial features should be incomplete."""
        partial = {"YearBuilt": 2005, "FullBath": 2}
        assert not is_complete(partial)

    def test_complete_with_all_features(self):
        """All features should be complete."""
        all_features = {
            "OverallQual": 7,
            "GrLivArea": 1850.0,
            "TotalBsmtSF": 900.0,
            "GarageCars": 2,
            "Neighborhood": "NAmes",
            "ExterQual": "Gd",
            "YearBuilt": 2005,
            "FullBath": 2,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 7,
            "LotArea": 8200.0,
        }
        assert is_complete(all_features)

    def test_complete_with_current_state(self):
        """Extraction + current state can achieve completeness."""
        current_state = {
            "OverallQual": 7,
            "GrLivArea": 1850.0,
            "TotalBsmtSF": 900.0,
            "GarageCars": 2,
            "Neighborhood": "NAmes",
            "ExterQual": "Gd",
            "YearBuilt": 2005,
            "FullBath": 2,
        }
        extracted = {
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 7,
            "LotArea": 8200.0,
        }
        assert is_complete(extracted, current_state)


class TestFollowUpFeatureSelection:
    """Test follow-up feature selection."""

    def test_pick_follow_up_prioritizes_numeric(self):
        """Should prioritize numeric features."""
        missing = [
            "OverallQual",
            "GrLivArea",
            "Neighborhood",
            "ExterQual",
        ]
        selected = pick_follow_up_features(missing, count=2)
        assert len(selected) == 2
        # Both selected should be numeric
        assert all(f in ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "FullBath", "TotRmsAbvGrd", "LotArea"] for f in selected)

    def test_pick_follow_up_respects_count(self):
        """Should return at most `count` features."""
        missing = REQUIRED_FEATURES.copy()
        selected = pick_follow_up_features(missing, count=3)
        assert len(selected) == 3

    def test_pick_follow_up_empty_missing(self):
        """Empty missing list should return empty result."""
        selected = pick_follow_up_features([])
        assert len(selected) == 0


class TestFollowUpMessage:
    """Test follow-up message generation."""

    def test_follow_up_message_for_missing_features(self):
        """Should generate natural message for missing features."""
        missing = ["OverallQual", "GrLivArea", "TotalBsmtSF"]
        msg = build_follow_up_message(missing)
        assert "overall quality" in msg.lower()
        assert "above-ground living area" in msg.lower()
        assert "basement" in msg.lower()

    def test_follow_up_message_all_missing(self):
        """Should ask for multiple features even if all missing."""
        missing = REQUIRED_FEATURES.copy()
        msg = build_follow_up_message(missing)
        assert len(msg) > 0
        assert "overall quality" in msg.lower()

    def test_follow_up_message_no_missing(self):
        """Should acknowledge completion for zero missing."""
        msg = build_follow_up_message([])
        assert "required" in msg.lower() or "complete" in msg.lower() or "ready" in msg.lower()

