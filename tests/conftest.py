"""Test fixtures and configuration."""
import pytest

from app.llm.client import GeminiClient, MockLLMClient
from app.models.feature_models import PropertyFeaturesPartial


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    return GeminiClient(use_mock=True)


@pytest.fixture
def mock_response_first_turn():
    """
    Mock response for first turn extraction.
    
    Based on stage1_eval_cases.yaml q1 case.
    """
    return """{
  "extracted_features": {
    "YearBuilt": 1998,
    "FullBath": 2,
    "GarageCars": 2,
    "KitchenQual": "Gd"
  },
  "missing_features": [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "Neighborhood",
    "ExterQual",
    "BsmtQual",
    "TotRmsAbvGrd",
    "LotArea"
  ],
  "is_complete": false,
  "assistant_message": "I still need the overall quality from 1 to 10, the above-ground living area in square feet, and the basement size in square feet."
}"""


@pytest.fixture
def mock_response_follow_up():
    """
    Mock response for follow-up turn.
    
    Based on stage1_eval_cases.yaml q2 case.
    """
    return """{
  "extracted_features": {
    "GrLivArea": 1850,
    "TotalBsmtSF": 900,
    "LotArea": 8200
  },
  "missing_features": [
    "OverallQual",
    "Neighborhood",
    "ExterQual",
    "BsmtQual",
    "TotRmsAbvGrd"
  ],
  "is_complete": false,
  "assistant_message": "I still need the overall quality from 1 to 10, the neighborhood, and the exterior quality."
}"""


@pytest.fixture
def mock_response_ordinal_stress():
    """
    Mock response for ordinal mapping stress test.
    
    Based on stage1_eval_cases.yaml q3 case.
    """
    return """{
  "extracted_features": {
    "ExterQual": "Ex",
    "KitchenQual": "Gd",
    "BsmtQual": "Fa"
  },
  "missing_features": [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "Neighborhood",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea"
  ],
  "is_complete": false,
  "assistant_message": "I still need the overall quality from 1 to 10, the year built, and the number of full bathrooms."
}"""


@pytest.fixture
def mock_response_vague():
    """
    Mock response for vague input (anti-hallucination).
    
    Based on stage1_eval_cases.yaml q4 case.
    """
    return """{
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
    "LotArea"
  ],
  "is_complete": false,
  "assistant_message": "I need a few concrete details to continue: the overall quality from 1 to 10, the above-ground living area in square feet, and the year built."
}"""


@pytest.fixture
def sample_partial_features():
    """Sample partial feature state for testing."""
    return PropertyFeaturesPartial(
        YearBuilt=2005,
        FullBath=2,
        GarageCars=1,
    )


@pytest.fixture
def sample_partial_features_complete():
    """Complete feature state for testing."""
    return PropertyFeaturesPartial(
        OverallQual=7,
        GrLivArea=1850.0,
        TotalBsmtSF=900.0,
        GarageCars=2,
        Neighborhood="NAmes",
        ExterQual="Gd",
        YearBuilt=2005,
        FullBath=2,
        KitchenQual="Gd",
        BsmtQual="TA",
        TotRmsAbvGrd=7,
        LotArea=8200.0,
    )

