"""Tests for the extraction normalizer."""
import pytest

from app.core.exceptions import NormalizationException
from app.extraction.normalizer import FeatureNormalizer


class TestNumericNormalization:
    """Test normalization of numeric features."""

    def test_normalize_integer(self):
        """Should convert string integers to int."""
        result = FeatureNormalizer.normalize_feature("GrLivArea", "3000")
        assert result == 3000
        assert isinstance(result, int)

    def test_normalize_float_to_int(self):
        """Should convert string integers without decimal."""
        result = FeatureNormalizer.normalize_feature("YearBuilt", "2005")
        assert result == 2005

    def test_normalize_float(self):
        """Should convert string floats to float."""
        result = FeatureNormalizer.normalize_feature("GrLivArea", "3000.5")
        assert result == 3000.5
        assert isinstance(result, float)

    def test_invalid_numeric(self):
        """Should raise exception on non-numeric string."""
        with pytest.raises(NormalizationException):
            FeatureNormalizer.normalize_feature("GrLivArea", "not a number")

    def test_numeric_none(self):
        """Should return None for None input."""
        result = FeatureNormalizer.normalize_feature("GrLivArea", None)
        assert result is None

    def test_numeric_empty_string(self):
        """Should return None for empty string."""
        result = FeatureNormalizer.normalize_feature("GrLivArea", "")
        assert result is None


class TestOrdinalNormalization:
    """Test normalization of ordinal features."""

    def test_ordinal_excellent(self):
        """Should map 'Excellent' to 'Ex'."""
        result = FeatureNormalizer.normalize_feature("ExterQual", "Excellent")
        assert result == "Ex"

    def test_ordinal_good(self):
        """Should map 'Good' to 'Gd'."""
        result = FeatureNormalizer.normalize_feature("KitchenQual", "Good")
        assert result == "Gd"

    def test_ordinal_typical(self):
        """Should map 'Typical' to 'TA'."""
        result = FeatureNormalizer.normalize_feature("BsmtQual", "Typical")
        assert result == "TA"

    def test_ordinal_typical_average(self):
        """Should map 'Typical/Average' to 'TA'."""
        result = FeatureNormalizer.normalize_feature("ExterQual", "Typical/Average")
        assert result == "TA"

    def test_ordinal_average(self):
        """Should map 'Average' to 'TA'."""
        result = FeatureNormalizer.normalize_feature("ExterQual", "Average")
        assert result == "TA"

    def test_ordinal_case_insensitive(self):
        """Should be case-insensitive."""
        result = FeatureNormalizer.normalize_feature("ExterQual", "excellent")
        assert result == "Ex"

    def test_ordinal_invalid(self):
        """Should raise exception on invalid ordinal value."""
        with pytest.raises(NormalizationException):
            FeatureNormalizer.normalize_feature("ExterQual", "Amazing")

    def test_ordinal_none_string(self):
        """Should map string 'None' to 'None'."""
        result = FeatureNormalizer.normalize_feature("BsmtQual", "None")
        assert result == "None"


class TestNominalNormalization:
    """Test normalization of nominal features."""

    def test_nominal_neighborhood(self):
        """Should return neighborhood as-is."""
        result = FeatureNormalizer.normalize_feature("Neighborhood", "Downtown")
        assert result == "Downtown"

    def test_nominal_with_spaces(self):
        """Should preserve neighborhood with spaces."""
        result = FeatureNormalizer.normalize_feature("Neighborhood", "North Ridge")
        assert result == "North Ridge"


class TestNormalizeAll:
    """Test batch normalization."""

    def test_normalize_all_mixed(self):
        """Should normalize a mix of features."""
        raw = {
            "GrLivArea": "3000",
            "FullBath": "2",
            "ExterQual": "Good",
            "Neighborhood": "Downtown",
            "YearBuilt": None,
        }
        result = FeatureNormalizer.normalize_all(raw)

        assert result["GrLivArea"] == 3000
        assert result["FullBath"] == 2
        assert result["ExterQual"] == "Gd"
        assert result["Neighborhood"] == "Downtown"
        assert result["YearBuilt"] is None

    def test_normalize_all_with_errors(self):
        """Should not fail on errors, just log them."""
        raw = {
            "GrLivArea": "invalid",  # This will cause an error
            "FullBath": "2",  # This is fine
        }
        result = FeatureNormalizer.normalize_all(raw)

        # GrLivArea should be set to None due to error
        assert result["GrLivArea"] is None
        assert result["FullBath"] == 2
