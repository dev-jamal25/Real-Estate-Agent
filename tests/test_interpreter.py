"""Tests for Stage 2 interpreter."""
import pytest
from unittest.mock import Mock

from app.llm.client import GeminiClient
from app.llm.interpreter import Stage2Interpreter
from app.models.feature_models import PropertyFeaturesComplete


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = Mock(spec=GeminiClient)
    mock.generate_stage1_response.return_value = (
        "This prediction is above the typical range. The large living area and excellent quality drive the higher price."
    )
    return mock


@pytest.fixture
def interpreter(mock_llm_client):
    """Create an interpreter with mock LLM client."""
    return Stage2Interpreter(llm_client=mock_llm_client, prompt_version="v1")


@pytest.fixture
def sample_features():
    """Create sample complete features."""
    return PropertyFeaturesComplete(
        OverallQual=8,
        GrLivArea=2000.0,
        TotalBsmtSF=1000.0,
        GarageCars=2,
        Neighborhood="CollgCr",
        ExterQual="Gd",
        YearBuilt=2005,
        FullBath=2,
        KitchenQual="Gd",
        BsmtQual="Gd",
        TotRmsAbvGrd=8,
        LotArea=8200.0,
    )


@pytest.fixture
def sample_global_stats():
    """Create sample global statistics."""
    return {
        "mean_price": 180921.0,
        "median_price": 163000.0,
        "min_price": 34900.0,
        "max_price": 755000.0,
        "q1_price": 129975.0,
        "q3_price": 235000.0,
        "typical_range": "129975.0 - 235000.0",
    }


class TestStage2Interpreter:
    """Test Stage 2 interpreter service."""

    def test_interpret_success(self, interpreter, sample_features, sample_global_stats):
        """Test successful interpretation."""
        predicted_price = 245000.0

        interpretation = interpreter.interpret(
            complete_features=sample_features,
            predicted_price=predicted_price,
            global_stats=sample_global_stats,
        )

        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert "above" in interpretation.lower() or "typical" in interpretation.lower()

    def test_interpret_calls_llm_with_correct_context(
        self, mock_llm_client, interpreter, sample_features, sample_global_stats
    ):
        """Test that LLM is called with correct context injected."""
        predicted_price = 245000.0

        interpreter.interpret(
            complete_features=sample_features,
            predicted_price=predicted_price,
            global_stats=sample_global_stats,
        )

        # Verify LLM was called once
        mock_llm_client.generate_stage1_response.assert_called_once()

        # Get the prompt that was passed
        call_args = mock_llm_client.generate_stage1_response.call_args
        prompt = call_args[0][0]

        # Verify context is injected
        assert "245,000.00" in prompt  # predicted_price formatted
        assert "180,921.00" in prompt  # global stats mean
        assert "2000" in prompt  # GrLivArea feature value
        assert "Gd" in prompt  # ExterQual feature value

    def test_interpret_with_missing_stats_uses_defaults(
        self, mock_llm_client, interpreter, sample_features
    ):
        """Test that missing global stats default to 0."""
        predicted_price = 245000.0
        incomplete_stats = {"target_mean": 180921.0}  # Other stats missing

        interpretation = interpreter.interpret(
            complete_features=sample_features,
            predicted_price=predicted_price,
            global_stats=incomplete_stats,
        )

        assert isinstance(interpretation, str)
        assert len(interpretation) > 0

    def test_interpret_returns_empty_response_error(self, interpreter, sample_features):
        """Test that empty LLM response raises ValueError."""
        mock_llm_client = Mock(spec=GeminiClient)
        mock_llm_client.generate_stage1_response.return_value = ""
        interpreter.llm_client = mock_llm_client

        with pytest.raises(ValueError, match="empty interpretation"):
            interpreter.interpret(
                complete_features=sample_features,
                predicted_price=245000.0,
                global_stats={"target_mean": 180921.0},
            )

    def test_interpret_strips_whitespace(self, mock_llm_client, interpreter, sample_features):
        """Test that interpretation response is stripped of extra whitespace."""
        mock_llm_client.generate_stage1_response.return_value = (
            "\n\n  This is an interpretation.  \n\n"
        )

        interpretation = interpreter.interpret(
            complete_features=sample_features,
            predicted_price=245000.0,
            global_stats={"target_mean": 180921.0},
        )

        assert interpretation == "This is an interpretation."
        assert not interpretation.startswith("\n")
        assert not interpretation.endswith("\n")
