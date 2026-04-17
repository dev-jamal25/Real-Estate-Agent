"""Tests for feature validation."""
import pytest

from app.core.exceptions import NormalizationException
from app.extraction.normalizer import FeatureNormalizer


class TestRangeDetection:
    """Test detection of ambiguous ranges and approximations."""
    
    def test_has_ambiguous_range_dash_pattern(self) -> None:
        """Test detection of dash-based ranges."""
        assert FeatureNormalizer._has_ambiguous_range("1-10")
        assert FeatureNormalizer._has_ambiguous_range("2000-2010")
        assert FeatureNormalizer._has_ambiguous_range("800-1000")
    
    def test_has_ambiguous_range_to_pattern(self) -> None:
        """Test detection of 'to' connector."""
        assert FeatureNormalizer._has_ambiguous_range("1 to 10")
        assert FeatureNormalizer._has_ambiguous_range("2000 to 2010")
    
    def test_has_ambiguous_range_between_and(self) -> None:
        """Test detection of 'between...and' pattern."""
        assert FeatureNormalizer._has_ambiguous_range("between 800 and 1000")
        assert FeatureNormalizer._has_ambiguous_range("Between 5 and 10")
    
    def test_has_ambiguous_range_around(self) -> None:
        """Test detection of 'around' approximation."""
        assert FeatureNormalizer._has_ambiguous_range("around 2000")
        assert FeatureNormalizer._has_ambiguous_range("Around 1500 sq ft")
        assert FeatureNormalizer._has_ambiguous_range("approximately 100")
        assert FeatureNormalizer._has_ambiguous_range("circa 1920")
    
    def test_has_ambiguous_range_or_pattern(self) -> None:
        """Test detection of 'or' pattern with numbers."""
        assert FeatureNormalizer._has_ambiguous_range("5 or 6")
        assert FeatureNormalizer._has_ambiguous_range("2000 or 2010")
    
    def test_has_ambiguous_range_negative_number(self) -> None:
        """Test that negative numbers are not flagged as ranges."""
        assert not FeatureNormalizer._has_ambiguous_range("-5")
        assert not FeatureNormalizer._has_ambiguous_range("-2000")
    
    def test_has_ambiguous_range_exact_values(self) -> None:
        """Test that exact values are not flagged as ranges."""
        assert not FeatureNormalizer._has_ambiguous_range("7")
        assert not FeatureNormalizer._has_ambiguous_range("2000")
        assert not FeatureNormalizer._has_ambiguous_range("1500.5")
        assert not FeatureNormalizer._has_ambiguous_range("Good")


class TestNumericNormalizationWithRangeDetection:
    """Test that numeric normalization properly rejects ambiguous ranges."""
    
    def test_normalize_numeric_exact_int(self) -> None:
        """Test exact integer values are accepted."""
        result = FeatureNormalizer.normalize_feature("OverallQual", "7")
        assert result == 7
    
    def test_normalize_numeric_exact_float(self) -> None:
        """Test exact float values are accepted."""
        result = FeatureNormalizer.normalize_feature("GrLivArea", "1500.5")
        assert result == 1500.5
    
    def test_normalize_numeric_rejects_dash_range(self) -> None:
        """Test that dash ranges are rejected."""
        with pytest.raises(NormalizationException, match="Ambiguous or range"):
            FeatureNormalizer.normalize_feature("OverallQual", "1-10")
    
    def test_normalize_numeric_rejects_to_pattern(self) -> None:
        """Test that 'to' patterns are rejected."""
        with pytest.raises(NormalizationException, match="Ambiguous or range"):
            FeatureNormalizer.normalize_feature("GrLivArea", "800 to 1000")
    
    def test_normalize_numeric_rejects_around_pattern(self) -> None:
        """Test that 'around' patterns are rejected."""
        with pytest.raises(NormalizationException, match="Ambiguous or range"):
            FeatureNormalizer.normalize_feature("YearBuilt", "around 2000")
    
    def test_normalize_numeric_rejects_between_pattern(self) -> None:
        """Test that 'between' patterns are rejected."""
        with pytest.raises(NormalizationException, match="Ambiguous or range"):
            FeatureNormalizer.normalize_feature("TotalBsmtSF", "between 800 and 1200")

