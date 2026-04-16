"""Tests for LLM feature extraction."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm.client import GeminiClient
from app.llm.extractor import Stage1Extractor
from app.models.feature_models import PropertyFeaturesPartial


class MockGeminiClientForTesting(GeminiClient):
    """Mock Gemini client that returns predefined responses."""

    def __init__(self, mock_response: str):
        """Initialize with a specific mock response."""
        super().__init__(use_mock=True)
        self.mock_response = mock_response

    def generate_stage1_response(self, prompt: str) -> str:
        """Return the predefined mock response."""
        return self.mock_response


class TestStage1ExtractorBasics:
    """Test basic extractor functionality."""

    def test_extractor_initialization(self):
        """Extractor should initialize with mock client by default."""
        extractor = Stage1Extractor()
        assert extractor.llm_client is not None
        assert extractor.prompt_version == "v1"

    def test_extractor_with_custom_client(self):
        """Extractor should accept custom LLM client."""
        custom_client = GeminiClient(use_mock=True)
        extractor = Stage1Extractor(llm_client=custom_client)
        assert extractor.llm_client is custom_client

    def test_extractor_with_v2_prompt(self):
        """Extractor should support v2 prompt."""
        extractor = Stage1Extractor(prompt_version="v2")
        assert extractor.prompt_version == "v2"


class TestStage1ExtractionFirstTurn:
    """Test extraction on initial user message (q1 from eval cases)."""

    def test_q1_partial_first_description(self, mock_response_first_turn):
        """Extract partial features from initial description."""
        query = "I want a house built in 1998 with 2 full bathrooms, a garage for 2 cars, and a good kitchen."

        mock_client = MockGeminiClientForTesting(mock_response_first_turn)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract(query)

        # Check extracted features
        assert result.extracted_features.YearBuilt == 1998
        assert result.extracted_features.FullBath == 2
        assert result.extracted_features.GarageCars == 2
        assert result.extracted_features.KitchenQual == "Gd"

        # Check missing features
        assert "OverallQual" in result.missing_features
        assert "GrLivArea" in result.missing_features
        assert len(result.missing_features) == 8

        # Not complete
        assert not result.is_complete

        # Has follow-up message
        assert len(result.assistant_message) > 0


class TestStage1ExtractionFollowUp:
    """Test extraction on follow-up turn (q2 from eval cases)."""

    def test_q2_follow_up_with_current_state(self, mock_response_follow_up):
        """Extract new features while using current state for completeness."""
        query = "The above-ground living area is 1850 square feet, the basement is 900 square feet, and the lot is around 8200 square feet."

        # Simulate current known state from previous turn
        current_state = PropertyFeaturesPartial(
            YearBuilt=1998,
            FullBath=2,
            GarageCars=2,
            KitchenQual="Gd",
        )

        mock_client = MockGeminiClientForTesting(mock_response_follow_up)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract(query, current_state)

        # Check newly extracted features
        assert result.extracted_features.GrLivArea == 1850.0
        assert result.extracted_features.TotalBsmtSF == 900.0
        assert result.extracted_features.LotArea == 8200.0

        # Previous features should NOT be in extracted_features
        assert result.extracted_features.YearBuilt is None
        assert result.extracted_features.FullBath is None

        # Missing should account for current state
        missing_set = set(result.missing_features)
        assert "YearBuilt" not in missing_set
        assert "FullBath" not in missing_set
        assert len(result.missing_features) == 5

        # Still incomplete
        assert not result.is_complete


class TestStage1ExtractionOrdinalMapping:
    """Test ordinal quality label mapping (q3 from eval cases)."""

    def test_q3_ordinal_mapping_stress_test(self, mock_response_ordinal_stress):
        """Map user-facing quality labels to model tokens."""
        query = "The exterior quality should be Excellent, the kitchen quality Good, and the basement quality Fair."

        mock_client = MockGeminiClientForTesting(mock_response_ordinal_stress)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract(query)

        # Check ordinal mapping
        assert result.extracted_features.ExterQual == "Ex"
        assert result.extracted_features.KitchenQual == "Gd"
        assert result.extracted_features.BsmtQual == "Fa"

        # Should have 9 missing (all except the 3 ordinals)
        assert len(result.missing_features) == 9
        assert "ExterQual" not in result.missing_features
        assert "KitchenQual" not in result.missing_features
        assert "BsmtQual" not in result.missing_features


class TestStage1ExtractionVagueInput:
    """Test hallucination resistance (q4 from eval cases)."""

    def test_q4_vague_non_usable_wording(self, mock_response_vague):
        """Vague wording should not produce extracted features."""
        query = "I want something beautiful, spacious, in a lovely area, and kind of old-fashioned."

        mock_client = MockGeminiClientForTesting(mock_response_vague)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract(query)

        # No features should be extracted from vague input
        extracted_dict = result.extracted_features.model_dump(exclude_none=True)
        assert len(extracted_dict) == 0

        # All features should be missing
        assert len(result.missing_features) == 12

        # Not complete
        assert not result.is_complete

        # Has follow-up asking for concrete details
        assert len(result.assistant_message) > 0


class TestStage1ExtractionInvalidJSON:
    """Test error handling for invalid LLM responses."""

    def test_invalid_json_response(self):
        """Invalid JSON should raise ValueError."""
        invalid_json = "This is not valid JSON at all"

        mock_client = MockGeminiClientForTesting(invalid_json)
        extractor = Stage1Extractor(llm_client=mock_client)

        with pytest.raises(ValueError, match="invalid JSON"):
            extractor.extract("some query")

    def test_missing_required_keys(self):
        """Missing required keys should raise ValueError."""
        incomplete_response = json.dumps(
            {
                "extracted_features": {},
                # Missing: missing_features, is_complete, assistant_message
            }
        )

        mock_client = MockGeminiClientForTesting(incomplete_response)
        extractor = Stage1Extractor(llm_client=mock_client)

        with pytest.raises(ValueError, match="missing required keys"):
            extractor.extract("some query")


class TestStage1ExtractionValueNormalization:
    """Test normalization of extracted values."""

    def test_numeric_normalization_in_extraction(self):
        """Numeric values should be coerced to correct types."""
        response = json.dumps(
            {
                "extracted_features": {
                    "OverallQual": "8",  # String should be coerced to int
                    "GrLivArea": 1850,  # Int should stay numeric
                    "YearBuilt": "2005",  # String should be coerced
                },
                "missing_features": [
                    "GrLivArea",
                    "TotalBsmtSF",
                    "GarageCars",
                    "Neighborhood",
                    "ExterQual",
                    "FullBath",
                    "KitchenQual",
                    "BsmtQual",
                    "TotRmsAbvGrd",
                    "LotArea",
                ],
                "is_complete": False,
                "assistant_message": "More info needed.",
            }
        )

        mock_client = MockGeminiClientForTesting(response)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract("query")

        # Check that values are normalized correctly
        assert result.extracted_features.OverallQual == 8.0
        assert result.extracted_features.GrLivArea == 1850.0
        assert result.extracted_features.YearBuilt == 2005.0

    def test_ordinal_normalization_in_extraction(self):
        """Ordinal values should be normalized to model tokens."""
        response = json.dumps(
            {
                "extracted_features": {
                    "ExterQual": "Gd",  # Already a model token
                    "KitchenQual": "Good",  # User-facing label
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
                "is_complete": False,
                "assistant_message": "More info needed.",
            }
        )

        mock_client = MockGeminiClientForTesting(response)
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract("query")

        # Both should end up as model tokens
        assert result.extracted_features.ExterQual == "Gd"
        assert result.extracted_features.KitchenQual == "Gd"

    def test_invalid_values_dropped_silently(self):
        """Invalid feature values should be dropped, not raise errors."""
        response = json.dumps(
            {
                "extracted_features": {
                    "OverallQual": "not_a_number",  # Invalid
                    "GrLivArea": 1850.0,  # Valid
                    "UnknownFeature": 123,  # Unknown feature
                },
                "missing_features": [
                    "OverallQual",
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
                "is_complete": False,
                "assistant_message": "More info needed.",
            }
        )

        mock_client = MockGeminiClientForTesting(response)
        extractor = Stage1Extractor(llm_client=mock_client)

        # Should not raise, just drop invalid values
        result = extractor.extract("query")

        # Only the valid value should be extracted
        extracted_dict = result.extracted_features.model_dump(exclude_none=True)
        assert len(extracted_dict) == 1
        assert extracted_dict["GrLivArea"] == 1850.0


class TestStage1ExtractionCompleteness:
    """Test completeness detection and accumulation."""

    def test_completeness_after_full_extraction(self, sample_partial_features_complete):
        """All features present should mark as complete."""
        response_json = {
            "extracted_features": sample_partial_features_complete.model_dump(),
            "missing_features": [],
            "is_complete": True,
            "assistant_message": "Ready to predict.",
        }

        mock_client = MockGeminiClientForTesting(json.dumps(response_json))
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract("query")

        assert result.is_complete
        assert len(result.missing_features) == 0
        assert "ready" in result.assistant_message.lower() or "complete" in result.assistant_message.lower()

    def test_completeness_with_accumulated_state(self, sample_partial_features_complete):
        """Completeness should account for previous state."""
        # Provide half the features in current state
        current_state = PropertyFeaturesPartial(
            OverallQual=7,
            GrLivArea=1850.0,
            TotalBsmtSF=900.0,
            GarageCars=2,
            Neighborhood="NAmes",
            ExterQual="Gd",
        )

        # Extract the other half
        extracted_dict = sample_partial_features_complete.model_dump()
        extracted_dict = {
            k: v for k, v in extracted_dict.items()
            if k in ["YearBuilt", "FullBath", "KitchenQual", "BsmtQual", "TotRmsAbvGrd", "LotArea"]
        }

        response_json = {
            "extracted_features": extracted_dict,
            "missing_features": [],
            "is_complete": True,
            "assistant_message": "Ready to predict.",
        }

        mock_client = MockGeminiClientForTesting(json.dumps(response_json))
        extractor = Stage1Extractor(llm_client=mock_client)

        result = extractor.extract("query", current_state)

        # Should be complete due to accumulated state
        assert result.is_complete
        assert len(result.missing_features) == 0


class TestStage1ExtractorPromptInjection:
    """Test that context is properly injected into prompts."""

    def test_prompt_receives_user_message(self):
        """User message should be injected into prompt."""
        # Create a mock that captures what it receives
        captured_prompt = None

        class CapturingMockClient(GeminiClient):
            def generate_stage1_response(self, prompt: str) -> str:
                nonlocal captured_prompt
                captured_prompt = prompt
                return json.dumps(
                    {
                        "extracted_features": {},
                        "missing_features": REQUIRED_FEATURES,
                        "is_complete": False,
                        "assistant_message": "Test",
                    }
                )

        user_msg = "This is my custom user query."
        extractor = Stage1Extractor(llm_client=CapturingMockClient(use_mock=True))

        extractor.extract(user_msg)

        # Verify user message was injected
        assert captured_prompt is not None
        assert user_msg in captured_prompt

    def test_prompt_receives_current_state(self):
        """Current feature state should be injected into prompt."""
        captured_prompt = None

        class CapturingMockClient(GeminiClient):
            def generate_stage1_response(self, prompt: str) -> str:
                nonlocal captured_prompt
                captured_prompt = prompt
                return json.dumps(
                    {
                        "extracted_features": {},
                        "missing_features": REQUIRED_FEATURES,
                        "is_complete": False,
                        "assistant_message": "Test",
                    }
                )

        current_state = PropertyFeaturesPartial(YearBuilt=2005, FullBath=2)
        extractor = Stage1Extractor(llm_client=CapturingMockClient(use_mock=True))

        extractor.extract("query", current_state)

        # Verify current state was injected
        assert captured_prompt is not None
        assert "2005" in captured_prompt
        assert "2" in captured_prompt


# Import required for inline tests
REQUIRED_FEATURES = [
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
]

