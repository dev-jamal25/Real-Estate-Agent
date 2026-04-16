"""Tests for Stage 1 extraction API endpoints."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.extraction.schemas import ExtractedFeatures, Stage1Response
from app.api.routes import router as api_router
from app.config import settings
from app.core.logging_config import configure_logging


@pytest.fixture
def app():
    """Create test app without lifespan for simpler testing."""
    configure_logging()
    
    # Create a simple FastAPI app without lifespan for testing
    test_app = FastAPI()
    test_app.include_router(api_router)
    
    return test_app


@pytest.fixture
def client(app):
    """Create test client with mocked dependency."""
    from app.api.dependencies import get_extraction_service
    from unittest.mock import MagicMock
    
    mock_service = MagicMock()
    app.dependency_overrides[get_extraction_service] = lambda: mock_service
    
    return TestClient(app), mock_service


@pytest.fixture
def mock_extraction_response():
    """Create a mock extraction response."""
    # Service returns only latest-turn extracted features (not merged state)
    return Stage1Response(
        extracted_features=ExtractedFeatures(
            YearBuilt=1998,
            FullBath=2,
            GarageCars=2,
            KitchenQual="Gd",
        ),
        missing_features=[
            "OverallQual",
            "GrLivArea",
            "TotalBsmtSF",
            "Neighborhood",
            "ExterQual",
            "BsmtQual",
            "TotRmsAbvGrd",
            "LotArea",
        ],
        is_complete=False,
        reply="I still need the overall quality, living area, and basement size.",
    )


class TestHealth:
    """Tests for GET /health endpoint."""

    def test_health_check(self, client):
        """Health check should return healthy status."""
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredict:
    """Tests for POST /predict endpoint."""

    def test_predict_valid_request(self, client, mock_extraction_response):
        """Valid prediction request should work."""
        test_client, mock_service = client
        request_data = {
            "query": "Built in 1998 with 2 full bathrooms, garage for 2 cars, good kitchen.",
            "accumulated_features": {},
        }

        mock_service.extract.return_value = mock_extraction_response

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Check response structure
        assert "accumulated_features" in result
        assert "extracted_features" in result
        assert "missing_features" in result
        assert "is_complete" in result
        assert "reply" in result

        # Check extracted features (only new ones)
        assert result["extracted_features"]["YearBuilt"] == 1998
        assert result["extracted_features"]["FullBath"] == 2
        assert result["extracted_features"]["GarageCars"] == 2
        assert result["extracted_features"]["KitchenQual"] == "Gd"

        # Check accumulated (merged state)
        assert result["accumulated_features"]["YearBuilt"] == 1998
        assert result["accumulated_features"]["FullBath"] == 2

        # Check missing features
        assert "OverallQual" in result["missing_features"]
        assert len(result["missing_features"]) == 8

        # Check completeness
        assert result["is_complete"] is False

        # Check reply
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0

    def test_predict_with_accumulated_state(self, client):
        """Should merge accumulated state with new extraction."""
        test_client, mock_service = client
        
        accumulated = {
            "YearBuilt": 1998,
            "FullBath": 2,
            "GarageCars": 2,
            "KitchenQual": "Gd",
        }

        request_data = {
            "query": "Living area is 1850 sq ft, basement is 900 sq ft.",
            "accumulated_features": accumulated,
        }

        response_data = Stage1Response(
            # Service returns only the NEW extracted features (latest-turn only)
            extracted_features=ExtractedFeatures(
                GrLivArea=1850,
                TotalBsmtSF=900,
            ),
            missing_features=["OverallQual", "Neighborhood", "ExterQual", "BsmtQual", "TotRmsAbvGrd", "LotArea"],
            is_complete=False,
            reply="Got it. I still need the overall quality, neighborhood, and exterior quality.",
        )

        mock_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Accumulated should include all merged features
        assert result["accumulated_features"]["YearBuilt"] == 1998
        assert result["accumulated_features"]["GrLivArea"] == 1850
        assert result["accumulated_features"]["TotalBsmtSF"] == 900

        # Extracted should only show new ones
        assert result["extracted_features"]["GrLivArea"] == 1850
        assert result["extracted_features"]["TotalBsmtSF"] == 900

        # Should not repeat old ones in extracted
        assert "YearBuilt" not in result["extracted_features"]  # Already in accumulated input

    def test_predict_empty_query_rejected(self, client):
        """Empty query should be rejected."""
        test_client, _ = client
        request_data = {
            "query": "",
            "accumulated_features": {},
        }

        response = test_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_query_rejected(self, client):
        """Missing query field should be rejected."""
        test_client, _ = client
        request_data = {
            "accumulated_features": {},
        }

        response = test_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_hallucination_resistance(self, client):
        """Should correctly handle vague input."""
        test_client, mock_service = client
        
        request_data = {
            "query": "I want something beautiful and spacious.",
            "accumulated_features": {},
        }

        response_data = Stage1Response(
            # Service returns empty - no new features extracted this turn
            extracted_features=ExtractedFeatures(),
            missing_features=[
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
            is_complete=False,
            reply="I need concrete details: overall quality from 1-10, living area in sq ft, and year built.",
        )

        mock_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # No features extracted
        assert len(result["extracted_features"]) == 0

        # All features still missing
        assert len(result["missing_features"]) == 12
        assert result["is_complete"] is False

    def test_predict_complete_extraction(self, client):
        """Should handle complete extraction."""
        test_client, mock_service = client
        
        all_features = {
            "OverallQual": 8,
            "GrLivArea": 2500,
            "TotalBsmtSF": 1200,
            "GarageCars": 3,
            "Neighborhood": "Downtown",
            "ExterQual": "Gd",
            "YearBuilt": 2000,
            "FullBath": 3,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 10,
            "LotArea": 20000,
        }

        request_data = {
            "query": "Provide all details...",
            "accumulated_features": all_features.copy(),
        }

        response_data = Stage1Response(
            # All features already accumulated, so service returns no new features
            extracted_features=ExtractedFeatures(),
            missing_features=[],
            is_complete=True,
            reply="Perfect! I have all the information I need.",
        )

        mock_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # All 12 features present
        assert len([v for v in result["accumulated_features"].values() if v is not None]) == 12

        # No missing features
        assert len(result["missing_features"]) == 0

        # Complete
        assert result["is_complete"] is True

    def test_predict_incomplete_no_prediction(self, client, mock_extraction_response):
        """Incomplete extraction should NOT call prediction service."""
        test_client, mock_extraction_service = client
        
        request_data = {
            "query": "Built in 1998 with 2 bathrooms",
            "accumulated_features": {},
        }

        mock_extraction_service.extract.return_value = mock_extraction_response

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Should be incomplete
        assert result["is_complete"] is False

        # Should NOT have prediction results
        assert result["predicted_price"] is None
        assert result["global_stats"] == {}

    def test_predict_complete_with_prediction(self, client):
        """Complete extraction should trigger prediction."""
        test_client, mock_extraction_service = client
        
        # Override prediction service as well
        from app.api.dependencies import get_prediction_service
        
        mock_prediction_service = MagicMock()
        mock_prediction_service.predict.return_value = MagicMock(
            predicted_price=275000.0,
            global_stats={"mean_price": 180000, "std_price": 80000},
        )
        
        test_client.app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

        all_features = {
            "OverallQual": 8,
            "GrLivArea": 2500,
            "TotalBsmtSF": 1200,
            "GarageCars": 3,
            "Neighborhood": "Downtown",
            "ExterQual": "Gd",
            "YearBuilt": 2000,
            "FullBath": 3,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 10,
            "LotArea": 20000,
        }

        request_data = {
            "query": "All details provided.",
            "accumulated_features": all_features.copy(),
        }

        response_data = Stage1Response(
            extracted_features=ExtractedFeatures(),
            missing_features=[],
            is_complete=True,
            reply="I have all the information. Let me estimate the price.",
        )

        mock_extraction_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Should be complete
        assert result["is_complete"] is True

        # Should have prediction results
        assert result["predicted_price"] == 275000.0
        assert result["global_stats"]["mean_price"] == 180000

        # Prediction service should have been called
        mock_prediction_service.predict.assert_called_once()

    def test_predict_complete_but_invalid_features(self, client):
        """Complete=True but invalid feature values should not crash."""
        test_client, mock_extraction_service = client
        
        from app.api.dependencies import get_prediction_service
        
        mock_prediction_service = MagicMock()
        test_client.app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

        # Invalid features - OverallQual too high
        invalid_features = {
            "OverallQual": 15,  # Invalid: > 10
            "GrLivArea": 2500,
            "TotalBsmtSF": 1200,
            "GarageCars": 3,
            "Neighborhood": "Downtown",
            "ExterQual": "Gd",
            "YearBuilt": 2000,
            "FullBath": 3,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 10,
            "LotArea": 20000,
        }

        request_data = {
            "query": "All details provided.",
            "accumulated_features": invalid_features,
        }

        response_data = Stage1Response(
            extracted_features=ExtractedFeatures(),
            missing_features=[],
            is_complete=True,
            reply="I have all the information.",
        )

        mock_extraction_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        # Should still return 200 (graceful degradation)
        assert response.status_code == 200
        result = response.json()

        # Should be marked complete but prediction should be skipped
        assert result["is_complete"] is True
        assert result["predicted_price"] is None
        assert result["global_stats"] == {}

        # Prediction service should NOT have been called
        mock_prediction_service.predict.assert_not_called()

    def test_predict_complete_with_interpretation(self, client):
        """Complete prediction should include interpretation."""
        test_client, mock_extraction_service = client
        
        # Override prediction and interpretation services
        from app.api.dependencies import get_prediction_service, get_interpretation_service
        
        mock_prediction_service = MagicMock()
        mock_prediction_service.predict.return_value = MagicMock(
            predicted_price=275000.0,
            global_stats={
                "median_price": 160100.0,
                "mean_price": 179194.5,
                "min_price": 12789.0,
                "max_price": 755000.0,
                "q1_price": 129425.0,
                "q3_price": 210000.0,
            },
        )
        
        mock_interpretation_service = MagicMock()
        mock_interpretation_service.interpret.return_value = (
            "Based on the provided details, this property is valued at $275,000, "
            "which is above the typical range. The high living area and excellent kitchen quality are value drivers."
        )
        
        test_client.app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
        test_client.app.dependency_overrides[get_interpretation_service] = lambda: mock_interpretation_service

        all_features = {
            "OverallQual": 8,
            "GrLivArea": 2500,
            "TotalBsmtSF": 1200,
            "GarageCars": 3,
            "Neighborhood": "Downtown",
            "ExterQual": "Gd",
            "YearBuilt": 2000,
            "FullBath": 3,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 10,
            "LotArea": 20000,
        }

        request_data = {
            "query": "All details provided.",
            "accumulated_features": all_features.copy(),
        }

        response_data = Stage1Response(
            extracted_features=ExtractedFeatures(),
            missing_features=[],
            is_complete=True,
            reply="I have all the information. Let me estimate the price.",
        )

        mock_extraction_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Should be complete
        assert result["is_complete"] is True

        # Should have prediction results
        assert result["predicted_price"] == 275000.0
        assert result["global_stats"]["median_price"] == 160100.0

        # Should have interpretation
        assert result["interpretation"] is not None
        assert "275,000" in result["interpretation"]
        assert len(result["interpretation"]) > 0

        # Both services should have been called
        mock_prediction_service.predict.assert_called_once()
        mock_interpretation_service.interpret.assert_called_once()

    def test_predict_incomplete_no_interpretation(self, client, mock_extraction_response):
        """Incomplete extraction should NOT call interpretation service."""
        test_client, mock_extraction_service = client
        
        from app.api.dependencies import get_interpretation_service
        
        mock_interpretation_service = MagicMock()
        test_client.app.dependency_overrides[get_interpretation_service] = lambda: mock_interpretation_service

        request_data = {
            "query": "Built in 1998 with 2 bathrooms",
            "accumulated_features": {},
        }

        mock_extraction_service.extract.return_value = mock_extraction_response

        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Should be incomplete
        assert result["is_complete"] is False

        # Should NOT have interpretation
        assert result["interpretation"] is None

        # Interpretation service should NOT have been called
        mock_interpretation_service.interpret.assert_not_called()

    def test_predict_interpretation_error_graceful_degradation(self, client):
        """Interpretation error should not fail the entire request."""
        test_client, mock_extraction_service = client
        
        from app.api.dependencies import get_prediction_service, get_interpretation_service
        from app.core.exceptions import LLMException
        
        mock_prediction_service = MagicMock()
        mock_prediction_service.predict.return_value = MagicMock(
            predicted_price=275000.0,
            global_stats={"median_price": 160100.0, "mean_price": 179194.5},
        )
        
        mock_interpretation_service = MagicMock()
        mock_interpretation_service.interpret.side_effect = LLMException("API error")
        
        test_client.app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
        test_client.app.dependency_overrides[get_interpretation_service] = lambda: mock_interpretation_service

        all_features = {
            "OverallQual": 8,
            "GrLivArea": 2500,
            "TotalBsmtSF": 1200,
            "GarageCars": 3,
            "Neighborhood": "Downtown",
            "ExterQual": "Gd",
            "YearBuilt": 2000,
            "FullBath": 3,
            "KitchenQual": "Gd",
            "BsmtQual": "TA",
            "TotRmsAbvGrd": 10,
            "LotArea": 20000,
        }

        request_data = {
            "query": "All details provided.",
            "accumulated_features": all_features.copy(),
        }

        response_data = Stage1Response(
            extracted_features=ExtractedFeatures(),
            missing_features=[],
            is_complete=True,
            reply="I have all information.",
        )

        mock_extraction_service.extract.return_value = response_data

        response = test_client.post("/predict", json=request_data)

        # Should still return 200 (graceful degradation)
        assert response.status_code == 200
        result = response.json()

        # Should have prediction but not interpretation
        assert result["is_complete"] is True
        assert result["predicted_price"] == 275000.0
        assert result["interpretation"] is None  # Failed to get interpretation
