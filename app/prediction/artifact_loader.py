"""Load and validate ML prediction artifact.

Loads the serialized model, features, metadata, and global stats
from artifacts/model.joblib.
"""
import logging
from pathlib import Path
from typing import Any, Dict

import joblib

from app.contracts.property_features import REQUIRED_FEATURES
from app.prediction.schemas import PredictionArtifact

logger = logging.getLogger(__name__)


def load_artifact(artifact_path: str) -> PredictionArtifact:
    """Load and validate prediction artifact.
    
    Args:
        artifact_path: Path to model.joblib file
        
    Returns:
        PredictionArtifact with validated model and metadata
        
    Raises:
        FileNotFoundError: If artifact file not found
        ValueError: If artifact structure is invalid or features don't match
    """
    path = Path(artifact_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    try:
        data = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load artifact from {artifact_path}: {e}") from e
    
    # Validate required keys
    required_keys = {"model", "features", "metadata", "global_stats"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Artifact missing required keys: {missing_keys}. "
            f"Has: {set(data.keys())}"
        )
    
    features = data.get("features", [])
    
    # Validate feature count
    if len(features) != len(REQUIRED_FEATURES):
        raise ValueError(
            f"Artifact has {len(features)} features, expected {len(REQUIRED_FEATURES)}"
        )
    
    # Validate feature names match
    expected_set = set(REQUIRED_FEATURES)
    artifact_set = set(features)
    if expected_set != artifact_set:
        missing = expected_set - artifact_set
        extra = artifact_set - expected_set
        raise ValueError(
            f"Feature mismatch. Missing: {missing}, Extra: {extra}"
        )
    
    logger.info(f"Loaded artifact from {artifact_path}")
    logger.debug(f"  Model type: {type(data['model']).__name__}")
    logger.debug(f"  Features: {features}")
    logger.debug(f"  Metadata keys: {list(data['metadata'].keys())}")
    logger.debug(f"  Global stats keys: {list(data['global_stats'].keys())}")
    
    return PredictionArtifact(
        model=data["model"],
        features=features,
        metadata=data.get("metadata", {}),
        global_stats=data.get("global_stats", {}),
    )
