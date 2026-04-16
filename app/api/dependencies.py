"""Dependency injection for API routes."""
import logging
from typing import Optional

from fastapi import Depends

from app.config import settings
from app.llm.client import GeminiClient
from app.llm.extractor import Stage1Extractor
from app.llm.interpreter import Stage2Interpreter
from app.ml.loader import ArtifactLoader, ModelArtifact
from app.ml.predictor import Predictor

logger = logging.getLogger(__name__)

# Global singletons (loaded on startup)
_artifact: Optional[ModelArtifact] = None
_predictor: Optional[Predictor] = None
_extractor: Optional[Stage1Extractor] = None
_interpreter: Optional[Stage2Interpreter] = None


def set_artifact(artifact: ModelArtifact) -> None:
    """Set the global artifact instance during startup."""
    global _artifact
    _artifact = artifact


def set_predictor(predictor: Predictor) -> None:
    """Set the global predictor instance during startup."""
    global _predictor
    _predictor = predictor


def set_extractor(extractor: Stage1Extractor) -> None:
    """Set the global extractor instance during startup."""
    global _extractor
    _extractor = extractor


def set_interpreter(interpreter: Stage2Interpreter) -> None:
    """Set the global interpreter instance during startup."""
    global _interpreter
    _interpreter = interpreter


def get_artifact() -> ModelArtifact:
    """Dependency: return loaded model artifact."""
    if _artifact is None:
        raise RuntimeError("Artifact not loaded during startup")
    return _artifact


def get_predictor() -> Predictor:
    """Dependency: return predictor service."""
    if _predictor is None:
        raise RuntimeError("Predictor not initialized during startup")
    return _predictor


def get_extractor() -> Stage1Extractor:
    """Dependency: return Stage 1 extractor service."""
    if _extractor is None:
        raise RuntimeError("Extractor not initialized during startup")
    return _extractor


def get_interpreter() -> Stage2Interpreter:
    """Dependency: return Stage 2 interpreter service."""
    if _interpreter is None:
        raise RuntimeError("Interpreter not initialized during startup")
    return _interpreter

