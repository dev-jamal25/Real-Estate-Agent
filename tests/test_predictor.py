"""Tests for ML prediction."""
import pytest
from unittest.mock import Mock, MagicMock

from app.ml.loader import ModelArtifact, ArtifactLoadError, ArtifactLoader
from app.ml.predictor import Predictor, PredictionError
from app.models.feature_models import PropertyFeaturesComplete


class TestArtifactLoader:
    """Test artifact loading and validation."""

    def test_artifact_loader_missing_file(self, tmp_path):
        """Test loader raises error when artifact file is missing."""
        missing_path = tmp_path / "missing.joblib"
        with pytest.raises(ArtifactLoadError, match="not found"):
            ArtifactLoader.load(missing_path)

    def test_artifact_loader_invalid_artifact(self, tmp_path):
        """Test loader raises error when artifact structure is invalid."""
        import joblib
        from sklearn.linear_model import LinearRegression

        # Create invalid artifact (missing required keys)
        invalid_artifact = {"model": LinearRegression()}
        artifact_path = tmp_path / "invalid.joblib"
        joblib.dump(invalid_artifact, artifact_path)

        with pytest.raises(ArtifactLoadError, match="missing required keys"):
            ArtifactLoader.load(artifact_path)

    def test_artifact_loader_missing_features(self, tmp_path):
        """Test loader raises error when required features are missing."""
        import joblib
        from sklearn.linear_model import LinearRegression

        # Create artifact with incomplete features list
        incomplete_artifact = {
            "model": LinearRegression(),
            "features": ["OverallQual", "GrLivArea"],  # Missing many required features
            "metadata": {},
            "global_stats": {},
        }
        artifact_path = tmp_path / "incomplete.joblib"
        joblib.dump(incomplete_artifact, artifact_path)

        with pytest.raises(ArtifactLoadError, match="missing required fields"):
            ArtifactLoader.load(artifact_path)

    def test_artifact_loader_valid_artifact(self, tmp_path):
        """Test loader successfully loads valid artifact."""
        import joblib
        from sklearn.linear_model import LinearRegression

        # Create valid artifact
        valid_artifact = {
            "model": LinearRegression(),
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
            "metadata": {"version": "1.0", "model_type": "LinearRegression"},
            "global_stats": {
                "target_mean": 180921.0,
                "target_std": 79442.5,
                "target_min": 34900.0,
                "target_max": 755000.0,
            },
        }
        artifact_path = tmp_path / "valid.joblib"
        joblib.dump(valid_artifact, artifact_path)

        # Load should succeed
        loaded = ArtifactLoader.load(artifact_path)
        assert isinstance(loaded, ModelArtifact)
        assert loaded.model is not None
        assert loaded.features == valid_artifact["features"]
        assert loaded.metadata["version"] == "1.0"


class TestPredictorInputOrdering:
    """Test predictor builds correct feature order for model."""

    def test_predictor_feature_order_matches_canonical(self):
        """Test predictor reorders features to match canonical order."""
        import joblib
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        import numpy as np

        # Create minimal mock model that tracks feature order
        received_features = []

        class TrackingModel:
            def predict(self, X):
                received_features.append(list(X.columns))
                return np.array([250000.0])

        pipeline = Pipeline([("model", TrackingModel())])

        artifact_dict = {
            "model": pipeline,
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
            "metadata": {},
            "global_stats": {},
        }
        artifact = ModelArtifact(artifact_dict, "test")
        predictor = Predictor(artifact)

        # Create features dict in random order
        features_complete = PropertyFeaturesComplete(
            LotArea=8200.0,
            GrLivArea=1850.0,
            TotalBsmtSF=900.0,
            YearBuilt=1998,
            FullBath=2,
            GarageCars=2,
            KitchenQual="Gd",
            OverallQual=8,
            ExterQual="Gd",
            Neighborhood="CollgCr",
            BsmtQual="Gd",
            TotRmsAbvGrd=8,
        )

        # Run prediction
        result = predictor.predict(features_complete)

        # Verify features were passed in canonical order
        assert len(received_features) == 1
        expected_order = [
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
        assert received_features[0] == expected_order

    def test_predictor_returns_float_prediction(self):
        """Test predictor returns float for price."""
        import numpy as np

        mock_model = Mock()
        mock_model.predict.return_value = np.array([245000.0])

        artifact_dict = {
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
            "metadata": {},
            "global_stats": {},
        }
        artifact = ModelArtifact(artifact_dict, "test")
        predictor = Predictor(artifact)

        features_complete = PropertyFeaturesComplete(
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
        )

        result = predictor.predict(features_complete)
        assert isinstance(result, float)
        assert result == 245000.0

    def test_predictor_error_on_model_failure(self):
        """Test predictor raises PredictionError if model fails."""
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Model error")

        artifact_dict = {
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
            "metadata": {},
            "global_stats": {},
        }
        artifact = ModelArtifact(artifact_dict, "test")
        predictor = Predictor(artifact)

        features_complete = PropertyFeaturesComplete(
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
        )

        with pytest.raises(PredictionError, match="Failed to run prediction"):
            predictor.predict(features_complete)

    def test_predictor_global_stats_access(self):
        """Test predictor exposes global stats from artifact."""
        mock_model = Mock()
        global_stats = {
            "target_mean": 180921.0,
            "target_std": 79442.5,
            "target_min": 34900.0,
            "target_max": 755000.0,
        }

        artifact_dict = {
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
            "metadata": {},
            "global_stats": global_stats,
        }
        artifact = ModelArtifact(artifact_dict, "test")
        predictor = Predictor(artifact)

        assert predictor.get_global_stats() == global_stats

