"""ML model prediction service."""
import logging
from typing import Any, Dict

import pandas as pd

from app.core.constants import REQUIRED_FEATURES
from app.ml.loader import ModelArtifact
from app.models.feature_models import PropertyFeaturesComplete

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails."""

    pass


class Predictor:
    """
    Service for running predictions on complete feature sets.
    
    Takes a complete, validated feature set and runs it through
    the ML model pipeline, returning the predicted sale price.
    """

    def __init__(self, artifact: ModelArtifact):
        """
        Initialize predictor with a loaded artifact.
        
        Args:
            artifact: ModelArtifact instance with fitted model
        """
        self.artifact = artifact
        self.model = artifact.model
        self.features = artifact.features

    def predict(self, features: PropertyFeaturesComplete) -> float:
        """
        Predict sale price for complete feature set.
        
        Args:
            features: PropertyFeaturesComplete with all 12 required fields
        
        Returns:
            Predicted sale price as float
        
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Convert features to dict in the exact order expected by the model
            feature_dict = features.model_dump(exclude_none=False)

            # Build DataFrame in canonical feature order
            df = pd.DataFrame([feature_dict])
            df = df[REQUIRED_FEATURES]  # Ensure correct column order

            logger.debug(
                f"Running prediction with features: {list(df.columns)}"
            )

            # Call model predict
            prediction = self.model.predict(df)[0]

            # Ensure prediction is float
            predicted_price = float(prediction)

            logger.debug(f"Prediction complete: ${predicted_price:,.2f}")

            return predicted_price

        except Exception as e:
            raise PredictionError(
                f"Failed to run prediction: {str(e)}"
            ) from e

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics from artifact.
        
        Returns:
            Dict with global stats (e.g., target mean, std, min, max)
        """
        return self.artifact.global_stats
