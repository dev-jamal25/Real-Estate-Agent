"""Tests for Stage 2 interpretation service."""
import pytest
from unittest.mock import MagicMock, patch

from app.core.exceptions import LLMException
from app.interpretation.service import InterpretationService


@pytest.fixture
def mock_openrouter_client():
    """Create a mock OpenRouter client."""
    return MagicMock()


@pytest.fixture
def interpretation_service(mock_openrouter_client):
    """Create an interpretation service with mock client."""
    service = InterpretationService(client=mock_openrouter_client, prompt_version="v1")
    return service, mock_openrouter_client


class TestInterpretationService:
    """Tests for InterpretationService."""

    def test_interpret_success(self, interpretation_service):
        """Test successful interpretation."""
        service, mock_client = interpretation_service
        
        # Mock LLM response
        expected_interpretation = (
            "Based on the provided details, this property is valued at $235,000, "
            "which is above the typical range of $180,000-$210,000. "
            "The high living area and excellent kitchen quality are likely the biggest value drivers."
        )
        mock_client.call_plain_text.return_value = expected_interpretation
        
        # Call interpret
        result = service.interpret(
            complete_features={
                "OverallQual": 7,
                "GrLivArea": 2500,
                "TotalBsmtSF": 1200,
                "GarageCars": 2,
                "Neighborhood": "NAmes",
                "ExterQual": "Gd",
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Gd",
                "BsmtQual": "Gd",
                "TotRmsAbvGrd": 10,
                "LotArea": 12000,
            },
            predicted_price=235000.0,
            global_stats={
                "median_price": 160100.0,
                "mean_price": 179194.5,
                "min_price": 12789.0,
                "max_price": 755000.0,
                "q1_price": 129425.0,
                "q3_price": 210000.0,
            },
        )
        
        # Verify result
        assert result == expected_interpretation
        mock_client.call_plain_text.assert_called_once()

    def test_interpret_empty_response(self, interpretation_service):
        """Test interpretation with empty LLM response."""
        service, mock_client = interpretation_service
        
        # Mock empty response
        mock_client.call_plain_text.return_value = "   \n\n   "
        
        # Verify exception
        with pytest.raises(LLMException, match="empty"):
            service.interpret(
                complete_features={
                    "OverallQual": 7,
                    "GrLivArea": 2500,
                    "TotalBsmtSF": 1200,
                    "GarageCars": 2,
                    "Neighborhood": "NAmes",
                    "ExterQual": "Gd",
                    "YearBuilt": 2005,
                    "FullBath": 2,
                    "KitchenQual": "Gd",
                    "BsmtQual": "Gd",
                    "TotRmsAbvGrd": 10,
                    "LotArea": 12000,
                },
                predicted_price=235000.0,
                global_stats={
                    "median_price": 160100.0,
                    "mean_price": 179194.5,
                    "min_price": 12789.0,
                    "max_price": 755000.0,
                    "q1_price": 129425.0,
                    "q3_price": 210000.0,
                },
            )

    def test_interpret_llm_exception(self, interpretation_service):
        """Test interpretation when LLM call fails."""
        service, mock_client = interpretation_service
        
        # Mock exception
        mock_client.call_plain_text.side_effect = LLMException("API key not set")
        
        # Verify exception is raised
        with pytest.raises(LLMException, match="Interpretation failed"):
            service.interpret(
                complete_features={
                    "OverallQual": 7,
                    "GrLivArea": 2500,
                    "TotalBsmtSF": 1200,
                    "GarageCars": 2,
                    "Neighborhood": "NAmes",
                    "ExterQual": "Gd",
                    "YearBuilt": 2005,
                    "FullBath": 2,
                    "KitchenQual": "Gd",
                    "BsmtQual": "Gd",
                    "TotRmsAbvGrd": 10,
                    "LotArea": 12000,
                },
                predicted_price=235000.0,
                global_stats={
                    "median_price": 160100.0,
                    "mean_price": 179194.5,
                    "min_price": 12789.0,
                    "max_price": 755000.0,
                    "q1_price": 129425.0,
                    "q3_price": 210000.0,
                },
            )

    def test_interpret_with_missing_stats(self, interpretation_service):
        """Test interpretation when some stats are missing."""
        service, mock_client = interpretation_service
        
        # Mock response
        expected_interpretation = "This is a test interpretation."
        mock_client.call_plain_text.return_value = expected_interpretation
        
        # Call with partial stats
        result = service.interpret(
            complete_features={
                "OverallQual": 7,
                "GrLivArea": 2500,
                "TotalBsmtSF": 1200,
                "GarageCars": 2,
                "Neighborhood": "NAmes",
                "ExterQual": "Gd",
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Gd",
                "BsmtQual": "Gd",
                "TotRmsAbvGrd": 10,
                "LotArea": 12000,
            },
            predicted_price=235000.0,
            global_stats={
                "median_price": 160100.0,
                "mean_price": 179194.5,
            },
        )
        
        # Verify result still works
        assert result == expected_interpretation
        
        # Verify call was made
        call_args = mock_client.call_plain_text.call_args
        assert call_args is not None
        system_prompt, user_context = call_args[0]
        # User context should contain "N/A" for missing stats
        assert "N/A" in user_context

    def test_build_user_context_formats_numbers(self, interpretation_service):
        """Test that user context properly formats numbers."""
        service, mock_client = interpretation_service
        
        mock_client.call_plain_text.return_value = "test"
        
        service.interpret(
            complete_features={
                "OverallQual": 7,
                "GrLivArea": 2500,
                "TotalBsmtSF": 1200,
                "GarageCars": 2,
                "Neighborhood": "NAmes",
                "ExterQual": "Gd",
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Gd",
                "BsmtQual": "Gd",
                "TotRmsAbvGrd": 10,
                "LotArea": 12000,
            },
            predicted_price=235000.50,
            global_stats={
                "median_price": 160100.0,
                "mean_price": 179194.47,
                "min_price": 12789.0,
                "max_price": 755000.0,
                "q1_price": 129425.0,
                "q3_price": 210000.0,
            },
        )
        
        # Verify context was formatted correctly
        call_args = mock_client.call_plain_text.call_args
        system_prompt, user_context = call_args[0]
        
        # Should contain formatted prices with commas and decimals
        assert "$235,000.50" in user_context
        assert "$160,100" in user_context
        assert "$179,194" in user_context
