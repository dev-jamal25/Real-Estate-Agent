"""Tests for Streamlit UI helper functions."""
import pytest
from unittest.mock import MagicMock, patch

from ui.streamlit_app import (
    check_backend_health,
    call_predict_api,
    reset_conversation,
    convert_user_value_to_backend,
)


class TestBackendHealth:
    """Tests for backend health check."""
    
    def test_health_check_success(self):
        """Health check should return True on 200 response."""
        with patch('ui.streamlit_app.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = check_backend_health()
            
            assert result is True
            mock_get.assert_called_once()
    
    def test_health_check_failure_non_200(self):
        """Health check should return False on non-200 response."""
        with patch('ui.streamlit_app.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            result = check_backend_health()
            
            assert result is False
    
    def test_health_check_connection_error(self):
        """Health check should return False on connection error."""
        with patch('ui.streamlit_app.requests.get') as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()
            
            result = check_backend_health()
            
            assert result is False


class TestPredictAPI:
    """Tests for /predict API calls."""
    
    def test_predict_api_success(self):
        """API call should return response on success."""
        with patch('ui.streamlit_app.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "is_complete": False,
                "reply": "Test reply",
                "accumulated_features": {"test": "value"},
            }
            mock_post.return_value = mock_response
            
            result = call_predict_api("test query", {})
            
            assert result is not None
            assert result["reply"] == "Test reply"
            mock_post.assert_called_once()
    
    def test_predict_api_failure_non_200(self):
        """API call should return None on non-200 response."""
        with patch('ui.streamlit_app.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            result = call_predict_api("test query", {})
            
            assert result is None
    
    def test_predict_api_connection_error(self):
        """API call should return None on connection error."""
        with patch('ui.streamlit_app.requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError()
            
            result = call_predict_api("test query", {})
            
            assert result is None
    
    def test_predict_api_invalid_json(self):
        """API call should return None on invalid JSON response."""
        with patch('ui.streamlit_app.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError()
            mock_post.return_value = mock_response
            
            result = call_predict_api("test query", {})
            
            assert result is None


class TestConvertUserValueToBackend:
    """Tests for converting user-friendly input to backend format."""
    
    def test_numeric_integer_features(self):
        """Integer numeric features should be converted to int."""
        assert convert_user_value_to_backend("OverallQual", 7) == 7
        assert convert_user_value_to_backend("OverallQual", "7") == 7
        assert convert_user_value_to_backend("GarageCars", 2) == 2
        assert convert_user_value_to_backend("FullBath", 2) == 2
        assert convert_user_value_to_backend("TotRmsAbvGrd", 8) == 8
        assert convert_user_value_to_backend("YearBuilt", 2000) == 2000
    
    def test_numeric_float_features(self):
        """Float numeric features should be converted to float."""
        result = convert_user_value_to_backend("GrLivArea", 1500.5)
        assert isinstance(result, (int, float))
        assert result == 1500.5
        
        result = convert_user_value_to_backend("TotalBsmtSF", 1000)
        assert isinstance(result, (int, float))
        assert result == 1000
    
    def test_ordinal_features(self):
        """Ordinal features should be converted from display text to model token."""
        assert convert_user_value_to_backend("ExterQual", "Good") == "Gd"
        assert convert_user_value_to_backend("ExterQual", "Fair") == "Fa"
        assert convert_user_value_to_backend("ExterQual", "Typical/Average") == "TA"
        assert convert_user_value_to_backend("ExterQual", "Excellent") == "Ex"
        assert convert_user_value_to_backend("ExterQual", "Poor") == "Po"
        assert convert_user_value_to_backend("ExterQual", "None") == "None"
        
        assert convert_user_value_to_backend("KitchenQual", "Good") == "Gd"
        assert convert_user_value_to_backend("BsmtQual", "Fair") == "Fa"
    
    def test_nominal_features(self):
        """Nominal features should be returned as string."""
        assert convert_user_value_to_backend("Neighborhood", "Downtown") == "Downtown"
        assert convert_user_value_to_backend("Neighborhood", "  Suburbs  ") == "Suburbs"
    
    def test_none_and_empty_values(self):
        """None and empty values should return None."""
        assert convert_user_value_to_backend("OverallQual", None) is None
        assert convert_user_value_to_backend("OverallQual", "") is None
        assert convert_user_value_to_backend("Neighborhood", None) is None

