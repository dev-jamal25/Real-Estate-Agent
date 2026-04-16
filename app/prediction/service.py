"""Prediction service for price estimation.

Accepts complete validated features, builds a DataFrame in model feature order,
and calls the model pipeline for prediction.
"""
import logging
from typing import Any, Dict

import pandas as pd

from app.prediction.schemas import CompletePropertyFeatures, PredictionArtifact, PredictionOutput

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making price predictions.
    
    Operates on validated complete features and produces predictions
    with global statistics for interpretation.
    """

    def __init__(self, artifact: PredictionArtifact):
        """Initialize prediction service with loaded artifact.
        
        Args:
            artifact: PredictionArtifact containing model, features, metadata, global_stats
        """
        self.model = artifact.model
        self.feature_order = artifact.features
        self.metadata = artifact.metadata
        self.global_stats = artifact.global_stats

        logger.debug(
            f"PredictionService initialized with {len(self.feature_order)} features"
        )

    def predict(self, features: CompletePropertyFeatures) -> PredictionOutput:
        """Make a price prediction from complete features.
        
        Args:
            features: CompletePropertyFeatures with all 12 required fields
            
        Returns:
            PredictionOutput with predicted_price and global_stats
            
        Raises:
            ValueError: If features cannot be converted to DataFrame or prediction fails
        """
        try:
            # Convert features to dict and build DataFrame in model feature order
            features_dict = features.model_dump()
            
            # Ensure we have values for all features in the correct order
            row_data = {}
            for feature_name in self.feature_order:
                if feature_name not in features_dict:
                    raise ValueError(f"Missing feature: {feature_name}")
                row_data[feature_name] = features_dict[feature_name]
            
            # Create single-row DataFrame in model feature order
            X = pd.DataFrame([row_data], columns=self.feature_order)
            
            logger.debug(
                f"Built DataFrame with shape {X.shape}, features: {list(X.columns)}"
            )
            
            # Run prediction
            prediction = self.model.predict(X)
            
            # Extract predicted price (assuming model returns array)
            predicted_price = float(prediction[0])
            
            logger.info(f"Prediction complete: ${predicted_price:.2f}")
            
            return PredictionOutput(
                predicted_price=predicted_price,
                global_stats=self.global_stats,
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e
