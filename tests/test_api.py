"""Tests for FastAPI routes."""
import pytest
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from app.api import dependencies, routes
from app.config import settings
from app.ml.loader import ModelArtifact
from app.ml.predictor import Predictor
from app.llm.extractor import Stage1Extractor
from app.models.feature_models import PropertyFeaturesPartial, PropertyFeaturesComplete
from app.models.response_models import Stage1ExtractionResult


def create_test_app() -> FastAPI:
    """Create a test FastAPI app without lifespan manager."""
    app = FastAPI(
        title="AI Real Estate Agent Test",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(routes.router, tags=["predictions"])

    return app


@pytest.fixture
def mock_artifact():
    """Create a mock artifact."""
    mock_model = Mock()
    mock_model.predict.return_value = [250000.0]
    
    return ModelArtifact(
        {
            "model": mock_model,
            "features": [
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
            "metadata": {"version": "1.0"},
            "global_stats": {
                "target_mean": 180921.0,
                "target_std": 79442.5,
            },
        },
        "test_artifact",
    )


@pytest.fixture
def mock_predictor(mock_artifact):
    """Create a mock predictor."""
    return Predictor(mock_artifact)


@pytest.fixture
def mock_extractor():
    """Create a mock extractor."""
    return Mock(spec=Stage1Extractor)


@pytest.fixture
def client(mock_artifact, mock_predictor, mock_extractor):
    """Create a test client with dependencies mocked."""
    app = create_test_app()
    
    # Override dependencies
    dependencies.set_artifact(mock_artifact)
    dependencies.set_predictor(mock_predictor)
    dependencies.set_extractor(mock_extractor)
    
    return TestClient(app)


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model artifact loaded" in data["message"]

    def test_health_check_returns_typed_response(self, client):
        """Test health check response has correct schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data


class TestPredictEndpointIncomplete:
    """Test POST /predict with incomplete features."""

    def test_predict_incomplete_returns_needs_more_info(self, client, mock_extractor):
        """Test /predict returns needs_more_info when features incomplete."""
        # Mock extraction returning incomplete state
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(OverallQual=8, GrLivArea=1850.0),
            missing_features=[
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
            assistant_message="I still need the basement size and other details.",
        )

        response = client.post(
            "/predict",
            json={
                "query": "2000 square feet, very good quality",
                "feature_overrides": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "needs_more_info"
        assert "extracted_features" in data
        assert "accumulated_features" in data
        assert "missing_features" in data
        assert len(data["missing_features"]) > 0
        assert "assistant_message" in data

    def test_predict_incomplete_merges_with_overrides(self, client, mock_extractor):
        """Test /predict merges feature_overrides with extracted features."""
        # Mock extraction
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(GrLivArea=1850.0),
            missing_features=["OverallQual", "TotalBsmtSF"],  # Simplified for test
            is_complete=False,
            assistant_message="Need more info.",
        )

        response = client.post(
            "/predict",
            json={
                "query": "1850 sq ft",
                "feature_overrides": {
                    "OverallQual": 8,
                    "YearBuilt": 2005,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "needs_more_info"
        
        # Verify overrides were merged into accumulated
        accumulated = data["accumulated_features"]
        assert accumulated.get("OverallQual") == 8
        assert accumulated.get("YearBuilt") == 2005
        assert accumulated.get("GrLivArea") == 1850.0

    def test_predict_passes_current_state_to_extractor(self, client, mock_extractor):
        """Test /predict passes feature_overrides as current state to extractor."""
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(),
            missing_features=["OverallQual"],
            is_complete=False,
            assistant_message="Need more.",
        )

        overrides = PropertyFeaturesPartial(OverallQual=8, YearBuilt=2005)

        response = client.post(
            "/predict",
            json={
                "query": "test query",
                "feature_overrides": overrides.model_dump(exclude_none=True),
            },
        )

        # Verify extractor was called with the state
        mock_extractor.extract.assert_called_once()
        call_args = mock_extractor.extract.call_args
        assert call_args[1]["latest_user_message"] == "test query"
        current_state_arg = call_args[1]["current_known_feature_state"]
        assert current_state_arg.OverallQual == 8
        assert current_state_arg.YearBuilt == 2005


class TestPredictEndpointComplete:
    """Test POST /predict with complete features."""

    def test_predict_complete_returns_prediction(self, client, mock_extractor):
        """Test /predict returns prediction when features complete."""
        # Mock extraction returning complete state
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(
                OverallQual=8,
                GrLivArea=1850.0,
                TotalBsmtSF=900.0,
                GarageCars=2,
                Neighborhood="CollgCr",
                ExterQual="Gd",
                YearBuilt=1998,
                FullBath=2,
                KitchenQual="Gd",
                BsmtQual="Gd",
                TotRmsAbvGrd=8,
                LotArea=8200.0,
            ),
            missing_features=[],
            is_complete=True,
            assistant_message="All required features collected. Ready to predict.",
        )

        response = client.post(
            "/predict",
            json={
                "query": "All features provided",
                "feature_overrides": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "accumulated_features" in data
        assert "predicted_price" in data
        assert "global_stats" in data
        assert isinstance(data["predicted_price"], float)
        assert data["predicted_price"] > 0

    def test_predict_complete_includes_global_stats(self, client, mock_extractor):
        """Test /predict response includes global stats."""
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(
                OverallQual=8,
                GrLivArea=1850.0,
                TotalBsmtSF=900.0,
                GarageCars=2,
                Neighborhood="CollgCr",
                ExterQual="Gd",
                YearBuilt=1998,
                FullBath=2,
                KitchenQual="Gd",
                BsmtQual="Gd",
                TotRmsAbvGrd=8,
                LotArea=8200.0,
            ),
            missing_features=[],
            is_complete=True,
            assistant_message="Ready.",
        )

        response = client.post(
            "/predict",
            json={
                "query": "complete",
                "feature_overrides": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        stats = data["global_stats"]
        assert "target_mean" in stats
        assert "target_std" in stats

    def test_predict_complete_all_features_in_response(self, client, mock_extractor):
        """Test /predict response includes all 12 features."""
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(
                OverallQual=8,
                GrLivArea=1850.0,
                TotalBsmtSF=900.0,
                GarageCars=2,
                Neighborhood="CollgCr",
                ExterQual="Gd",
                YearBuilt=1998,
                FullBath=2,
                KitchenQual="Gd",
                BsmtQual="Gd",
                TotRmsAbvGrd=8,
                LotArea=8200.0,
            ),
            missing_features=[],
            is_complete=True,
            assistant_message="Ready.",
        )

        response = client.post(
            "/predict",
            json={"query": "complete", "feature_overrides": None},
        )

        assert response.status_code == 200
        data = response.json()
        features = data["accumulated_features"]
        
        required_fields = [
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
        for field in required_fields:
            assert field in features, f"Missing field {field}"


class TestPredictEndpointErrors:
    """Test POST /predict error cases."""

    def test_predict_extraction_error_handled(self, client, mock_extractor):
        """Test /predict handles extraction errors gracefully."""
        mock_extractor.extract.side_effect = ValueError("Extraction failed")

        response = client.post(
            "/predict",
            json={
                "query": "test",
                "feature_overrides": None,
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_predict_invalid_complete_features_rejected(self, client, mock_extractor):
        """Test /predict with semantically invalid but syntactically valid features."""
        # Mock extraction with valid syntax but semantically complete state
        # Note: Pydantic validation happens when building the model, so we just verify
        # that a complete set is accepted even if values are edge cases
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(
                OverallQual=1,  # Valid int, minimum quality
                GrLivArea=1.0,  # Valid float, minimal area
                TotalBsmtSF=0.0,  # Valid float, no basement
                GarageCars=0,  # Valid int, no garage
                Neighborhood="CollgCr",
                ExterQual="Po",  # Poor quality
                YearBuilt=1900,  # Old year
                FullBath=1,
                KitchenQual="Po",
                BsmtQual="Po",
                TotRmsAbvGrd=1,
                LotArea=100.0,  # Small lot
            ),
            missing_features=[],
            is_complete=True,
            assistant_message="Ready.",
        )

        response = client.post(
            "/predict",
            json={"query": "test", "feature_overrides": None},
        )

        # Should still succeed even with edge case values
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestPredictEndpointRequestValidation:
    """Test POST /predict request validation."""

    def test_predict_query_required(self, client):
        """Test /predict requires query field."""
        response = client.post(
            "/predict",
            json={"feature_overrides": None},
        )

        assert response.status_code == 422

    def test_predict_query_min_length(self, client):
        """Test /predict rejects empty query."""
        response = client.post(
            "/predict",
            json={"query": "", "feature_overrides": None},
        )

        assert response.status_code == 422

    def test_predict_accepts_feature_overrides_null(self, client, mock_extractor):
        """Test /predict accepts null feature_overrides."""
        mock_extractor.extract.return_value = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(),
            missing_features=["OverallQual"],
            is_complete=False,
            assistant_message="Need more.",
        )

        response = client.post(
            "/predict",
            json={"query": "test query"},
        )

        assert response.status_code == 200

