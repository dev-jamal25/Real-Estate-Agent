"""Tests for the extraction service."""
from unittest.mock import MagicMock, patch

import pytest

from app.extraction.schemas import Stage1Response
from app.extraction.service import ExtractionService
from app.llm.openrouter_client import OpenRouterClient


class MockOpenRouterClient(OpenRouterClient):
    """Mock client that returns predefined responses."""

    def __init__(self, response_dict):
        """Initialize with mock response."""
        self.response_dict = response_dict
        self.api_key = "mock-key"

    def call(self, system_prompt, user_message):
        """Return mock response."""
        return self.response_dict


class TestExtractionService:
    """Test the main extraction service."""

    def test_extract_basic(self):
        """Should extract features from user message."""
        mock_response = {
            "extracted_features": {"GrLivArea": 3000, "FullBath": 2},
            "missing_features": [
                "OverallQual",
                "TotalBsmtSF",
                "GarageCars",
                "Neighborhood",
                "ExterQual",
                "YearBuilt",
                "KitchenQual",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "Got it. What about the overall quality and year built?",
        }

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)
        result = service.extract("3000 sq ft, 2 bathrooms")

        assert isinstance(result, Stage1Response)
        assert result.extracted_features.GrLivArea == 3000
        assert result.extracted_features.FullBath == 2
        assert result.is_complete is False
        assert len(result.missing_features) == 10
        assert result.reply == "Got it. What about the overall quality and year built?"

    def test_extract_complete(self):
        """Should mark as complete when all features present."""
        mock_response = {
            "extracted_features": {
                "OverallQual": 8,
                "GrLivArea": 3000,
                "TotalBsmtSF": 1000,
                "GarageCars": 3,
                "Neighborhood": "Downtown",
                "ExterQual": "Good",
                "YearBuilt": 2005,
                "FullBath": 2,
                "KitchenQual": "Good",
                "BsmtQual": "Typical",
                "TotRmsAbvGrd": 8,
                "LotArea": 15000,
            },
            "missing_features": [],
            "reply": "All set!",
        }

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)
        result = service.extract("Full property details...")

        assert result.is_complete is True
        assert result.missing_features == []

    def test_extract_with_accumulated_state(self):
        """Should merge with accumulated state."""
        accumulated = {
            "GrLivArea": 3000,
            "FullBath": 2,
            "YearBuilt": None,
        }

        mock_response = {
            "extracted_features": {"YearBuilt": 2005},
            "missing_features": [
                "OverallQual",
                "TotalBsmtSF",
                "GarageCars",
                "Neighborhood",
                "ExterQual",
                "KitchenQual",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "Got the year. What's the garage size?",
        }

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)
        result = service.extract("Built in 2005", accumulated_state=accumulated)

        # Previously known values should be preserved
        assert result.extracted_features.GrLivArea == 3000
        assert result.extracted_features.FullBath == 2
        # New value from this turn
        assert result.extracted_features.YearBuilt == 2005

    def test_extract_normalizes_ordinal(self):
        """Should normalize ordinal values to model tokens."""
        mock_response = {
            "extracted_features": {
                "ExterQual": "Good",
                "KitchenQual": "Excellent",
            },
            "missing_features": [
                "OverallQual",
                "GrLivArea",
                "TotalBsmtSF",
                "GarageCars",
                "Neighborhood",
                "YearBuilt",
                "FullBath",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "Noted. Anything else?",
        }

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)
        result = service.extract("Good exterior, excellent kitchen")

        # Should be normalized to model tokens
        assert result.extracted_features.ExterQual == "Gd"
        assert result.extracted_features.KitchenQual == "Ex"

    def test_extract_empty_message(self):
        """Should handle empty extraction gracefully."""
        mock_response = {
            "extracted_features": {},
            "missing_features": [
                "OverallQual",
                "GrLivArea",
                "TotalBsmtSF",
                "GarageCars",
                "Neighborhood",
                "ExterQual",
                "YearBuilt",
                "FullBath",
                "KitchenQual",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "I didn't catch any property details. Can you tell me about the house?",
        }

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)
        result = service.extract("What?")

        assert result.is_complete is False
        assert len(result.missing_features) == 12

    def test_extract_prompt_version(self):
        """Should use specified prompt version."""
        mock_response = {
            "extracted_features": {},
            "missing_features": ["OverallQual", "GrLivArea"],
            "reply": "Tell me more",
        }

        # Mock the prompt loader to capture which version was requested
        with patch("app.extraction.service.load_extract_prompt") as mock_loader:
            mock_loader.return_value = "mock prompt v2"

            client = MockOpenRouterClient(mock_response)
            service = ExtractionService(client=client, prompt_version="v2")
            service.extract("Some message")

            # Should request v2
            mock_loader.assert_called_once_with("v2")

    def test_extract_invalid_response_structure(self):
        """Should raise exception on invalid LLM response."""
        from app.core.exceptions import ExtractionException

        # Missing required field
        mock_response = {"extracted_features": {}}

        client = MockOpenRouterClient(mock_response)
        service = ExtractionService(client=client)

        with pytest.raises(ExtractionException):
            service.extract("Some message")

    @patch("app.extraction.service.OpenRouterClient")
    def test_extract_llm_call_failure(self, mock_client_class):
        """Should wrap LLM call failures."""
        from app.core.exceptions import ExtractionException

        mock_client = MagicMock()
        mock_client.call.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        service = ExtractionService(client=mock_client)

        with pytest.raises(ExtractionException):
            service.extract("Some message")


class TestExtractionServiceIntegration:
    """Integration tests (with mock client)."""

    def test_full_conversation_flow(self):
        """Should handle multi-turn conversation."""
        # Turn 1
        turn1_response = {
            "extracted_features": {"GrLivArea": 3000, "FullBath": 2},
            "missing_features": [
                "OverallQual",
                "TotalBsmtSF",
                "GarageCars",
                "Neighborhood",
                "ExterQual",
                "YearBuilt",
                "KitchenQual",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "OK. What about the garage and basement?",
        }

        client1 = MockOpenRouterClient(turn1_response)
        service1 = ExtractionService(client=client1)
        result1 = service1.extract("3000 sq ft, 2 full baths")

        assert result1.is_complete is False

        # Turn 2 - use result1 state as input
        turn2_response = {
            "extracted_features": {"GarageCars": 2, "TotalBsmtSF": 1000},
            "missing_features": [
                "OverallQual",
                "Neighborhood",
                "ExterQual",
                "YearBuilt",
                "KitchenQual",
                "BsmtQual",
                "TotRmsAbvGrd",
                "LotArea",
            ],
            "reply": "Got it. Any other details?",
        }

        accumulated = {
            "GrLivArea": result1.extracted_features.GrLivArea,
            "FullBath": result1.extracted_features.FullBath,
        }

        client2 = MockOpenRouterClient(turn2_response)
        service2 = ExtractionService(client=client2)
        result2 = service2.extract("2-car garage, 1000 sq ft basement", accumulated)

        assert result2.extracted_features.GrLivArea == 3000
        assert result2.extracted_features.GarageCars == 2
        assert result2.extracted_features.TotalBsmtSF == 1000
