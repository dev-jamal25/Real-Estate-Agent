"""Tests for display mapping functionality."""
import pytest

from app.core.display_mapping import (
    format_feature_summary,
    get_feature_display_name,
    get_feature_display_value,
    get_fallback_reply,
)


class TestDisplayNames:
    """Test feature name display mapping."""
    
    def test_get_feature_display_name_known_features(self) -> None:
        """Test that known features are mapped to friendly names."""
        assert get_feature_display_name("OverallQual") == "Overall quality"
        assert get_feature_display_name("GrLivArea") == "Above-ground living area"
        assert get_feature_display_name("TotalBsmtSF") == "Total basement area"
        assert get_feature_display_name("GarageCars") == "Garage spaces"
        assert get_feature_display_name("FullBath") == "Full bathrooms"
    
    def test_get_feature_display_name_unknown_feature(self) -> None:
        """Test that unknown features return original name."""
        assert get_feature_display_name("UnknownFeature") == "UnknownFeature"


class TestDisplayValues:
    """Test feature value formatting."""
    
    def test_get_feature_display_value_none(self) -> None:
        """Test None values display as 'Not specified'."""
        assert get_feature_display_value("GrLivArea", None) == "Not specified"
    
    def test_get_feature_display_value_ordinal_exterior_quality(self) -> None:
        """Test ordinal value for ExterQual."""
        assert get_feature_display_value("ExterQual", "Gd") == "Good"
        assert get_feature_display_value("ExterQual", "TA") == "Typical/Average"
        assert get_feature_display_value("ExterQual", "Fa") == "Fair"
        assert get_feature_display_value("ExterQual", "Ex") == "Excellent"
        assert get_feature_display_value("ExterQual", "Po") == "Poor"
    
    def test_get_feature_display_value_overall_quality(self) -> None:
        """Test OverallQual is formatted as x/10 scale."""
        assert get_feature_display_value("OverallQual", 7) == "7/10"
        assert get_feature_display_value("OverallQual", 10) == "10/10"
        assert get_feature_display_value("OverallQual", 1) == "1/10"
    
    def test_get_feature_display_value_numeric_with_units(self) -> None:
        """Test numeric features with units are formatted with commas."""
        assert get_feature_display_value("GrLivArea", 1500) == "1,500 sq ft"
        assert get_feature_display_value("TotalBsmtSF", 2000) == "2,000 sq ft"
        assert get_feature_display_value("LotArea", 10000) == "10,000 sq ft"
    
    def test_get_feature_display_value_year(self) -> None:
        """Test YearBuilt is formatted as plain integer."""
        assert get_feature_display_value("YearBuilt", 2000) == "2000"
        assert get_feature_display_value("YearBuilt", 1950) == "1950"
    
    def test_get_feature_display_value_counts(self) -> None:
        """Test count features are formatted as plain integers."""
        assert get_feature_display_value("FullBath", 2) == "2"
        assert get_feature_display_value("TotRmsAbvGrd", 8) == "8"
        assert get_feature_display_value("GarageCars", 3) == "3"
    
    def test_get_feature_display_value_nominal(self) -> None:
        """Test nominal feature (Neighborhood) returns as string."""
        assert get_feature_display_value("Neighborhood", "Downtown") == "Downtown"
        assert get_feature_display_value("Neighborhood", "Suburbs") == "Suburbs"


class TestFormatFeatureSummary:
    """Test the feature summary formatting function."""
    
    def test_format_feature_summary_empty_dict(self) -> None:
        """Test empty features dict."""
        result = format_feature_summary({})
        assert result == "(No features captured yet)"
    
    def test_format_feature_summary_with_features(self) -> None:
        """Test formatting a dict of features."""
        features = {
            "OverallQual": 7,
            "GrLivArea": 1500,
            "GarageCars": 2,
        }
        result = format_feature_summary(features)
        # Should be sorted and use display names and values
        assert "**Above-ground living area:** 1,500 sq ft" in result
        assert "**Garage spaces:** 2" in result
        assert "**Overall quality:** 7/10" in result
    
    def test_format_feature_summary_with_ordinal(self) -> None:
        """Test formatting with ordinal features."""
        features = {
            "ExterQual": "Gd",
            "FullBath": 2,
        }
        result = format_feature_summary(features)
        assert "**Exterior quality:** Good" in result
        assert "**Full bathrooms:** 2" in result


class TestFallbackReply:
    """Test the fallback reply message."""
    
    def test_get_fallback_reply(self) -> None:
        """Test fallback reply is a non-empty string."""
        reply = get_fallback_reply()
        assert isinstance(reply, str)
        assert len(reply) > 0
        # Should not contain raw internal jargon
        assert "(No reply" not in reply
