"""Custom exception classes."""


class ExtractionException(Exception):
    """Base exception for extraction errors."""

    pass


class LLMException(ExtractionException):
    """Exception raised when LLM call fails."""

    pass


class PromptLoadException(ExtractionException):
    """Exception raised when prompt loading fails."""

    pass


class ValidationException(ExtractionException):
    """Exception raised when validation of extracted values fails."""

    pass


class NormalizationException(ExtractionException):
    """Exception raised when normalization of extracted values fails."""

    pass

