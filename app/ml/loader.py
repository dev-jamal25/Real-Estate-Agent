"""Artifact loader: safely loads and validates the ML model artifact."""
import logging
from pathlib import Path
from typing import Any, Dict

import joblib

from app.core.constants import REQUIRED_FEATURES

logger = logging.getLogger(__name__)


class ArtifactLoadError(Exception):
    """Raised when artifact loading or validation fails."""

    pass


class ModelArtifact:
    """
    Wrapper for loaded ML model artifact.
    
    Ensures the artifact has all required keys and validates the model
    is compatible with the canonical feature contract.
    """

    def __init__(self, artifact: Dict[str, Any], artifact_path: str):
        """
        Initialize artifact wrapper and validate structure.
        
        Args:
            artifact: Loaded artifact dict
            artifact_path: Path to the artifact file (for logging)
        
        Raises:
            ArtifactLoadError: If artifact is missing required keys or invalid
        """
        self.artifact_path = artifact_path
        self._validate_artifact(artifact)
        self._artifact = artifact

    @staticmethod
    def _validate_artifact(artifact: Dict[str, Any]) -> None:
        """
        Validate artifact has all required keys and correct structure.
        
        Args:
            artifact: Artifact dict to validate
        
        Raises:
            ArtifactLoadError: If validation fails
        """
        required_keys = {"model", "features", "metadata", "global_stats"}
        missing_keys = required_keys - set(artifact.keys())
        if missing_keys:
            raise ArtifactLoadError(
                f"Artifact missing required keys: {missing_keys}. "
                f"Has keys: {set(artifact.keys())}"
            )

        # Validate artifact features match or contain required features
        artifact_features = artifact.get("features", [])
        if not artifact_features:
            raise ArtifactLoadError("Artifact 'features' list is empty")

        # Check that canonical features are present in artifact
        artifact_features_set = set(artifact_features)
        required_set = set(REQUIRED_FEATURES)
        missing_in_artifact = required_set - artifact_features_set
        if missing_in_artifact:
            raise ArtifactLoadError(
                f"Artifact features missing required fields: {missing_in_artifact}. "
                f"Artifact has: {artifact_features}"
            )

        logger.debug(
            f"Artifact validation passed. Features: {len(artifact_features)}, "
            f"includes all {len(REQUIRED_FEATURES)} required features"
        )

    @property
    def model(self) -> Any:
        """Get the fitted model pipeline."""
        return self._artifact["model"]

    @property
    def features(self) -> list:
        """Get the ordered list of feature names."""
        return self._artifact["features"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self._artifact.get("metadata", {})

    @property
    def global_stats(self) -> Dict[str, Any]:
        """Get global statistics (e.g., target mean, std)."""
        return self._artifact.get("global_stats", {})


class ArtifactLoader:
    """Singleton-like loader for ML model artifact."""

    _instance: ModelArtifact | None = None

    @staticmethod
    def load(artifact_path: str | Path) -> ModelArtifact:
        """
        Load artifact from disk.
        
        Loads the artifact only once and caches it. Subsequent calls
        return the cached instance.
        
        Args:
            artifact_path: Path to the artifact.joblib file
        
        Returns:
            ModelArtifact instance
        
        Raises:
            ArtifactLoadError: If file not found or artifact is invalid
        """
        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise ArtifactLoadError(
                f"Artifact file not found: {artifact_path.absolute()}"
            )

        try:
            logger.info(f"Loading artifact from {artifact_path}")
            raw_artifact = joblib.load(artifact_path)
            artifact = ModelArtifact(raw_artifact, str(artifact_path))
            logger.info(f"Artifact loaded successfully from {artifact_path}")
            return artifact
        except Exception as e:
            raise ArtifactLoadError(
                f"Failed to load or validate artifact from {artifact_path}: {str(e)}"
            ) from e
