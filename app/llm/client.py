"""LLM client wrapper for Gemini API calls with optional local Ollama fallback."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Exceptions for fallback-allowed provider errors
class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ProviderQuotaError(ProviderError):
    """Provider hit quota or rate limit."""

    pass


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    pass


class ProviderServerError(ProviderError):
    """Provider returned 5xx or temporary service error."""

    pass


class ProviderUnavailableError(ProviderError):
    """Provider is unavailable or unreachable."""

    pass


class MockLLMClient:
    """
    Mock LLM client for testing and local development.

    Returns deterministic JSON responses for predictable testing.
    """

    def generate_stage1_response(self, prompt: str) -> str:
        """
        Generate a mock Stage 1 extraction response.

        Returns valid JSON with a structured extraction.
        """
        # For testing purposes, return a simple empty/partial extraction
        response = {
            "extracted_features": {},
            "missing_features": [],
            "is_complete": False,
            "assistant_message": "Please provide more details about the property.",
        }
        return json.dumps(response)


class OllamaClient:
    """
    Local Ollama client for Stage 1 extraction fallback.

    Intended for local development and testing only.
    Requires a running Ollama instance at the configured endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 30,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama base URL (default: http://localhost:11434)
            model: Model name to use (default: llama3.2:1b)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_stage1_response(self, prompt: str) -> str:
        """
        Generate a Stage 1 extraction response using local Ollama.

        Args:
            prompt: Full prompt with context injected

        Returns:
            JSON string with extraction result

        Raises:
            ProviderUnavailableError: If Ollama is unreachable
            ProviderTimeoutError: If request times out
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package not installed. Install it with: pip install requests"
            )

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            response = requests.post(url, json=payload, timeout=self.timeout)

            # Check for HTTP errors
            if response.status_code >= 500:
                logger.error(
                    f"Ollama server error {response.status_code}: {response.text}"
                )
                raise ProviderServerError(f"Ollama returned {response.status_code}")

            response.raise_for_status()

            # Parse Ollama response
            data = response.json()
            return data.get("response", "")

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            raise ProviderTimeoutError("Ollama request timed out")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot reach Ollama at {self.base_url}: {e}")
            raise ProviderUnavailableError(f"Ollama unreachable at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise ProviderUnavailableError(f"Ollama request failed: {e}")


class GeminiClient:

    def __init__(
        self,
        use_mock: bool,
        api_key: str,
        model_name: str,
        enable_fallback: bool,
        fallback_on_quota: bool,
        fallback_on_timeout: bool,
        fallback_on_5xx: bool,
        ollama_enabled: bool,
        ollama_base_url: str,
        ollama_model: str,
        ollama_timeout: int,
    ):
        
        self.use_mock = use_mock
        self.api_key = api_key
        self.model_name = model_name

        # Fallback configuration (dev/local only)
        self.enable_fallback = enable_fallback
        self.fallback_on_quota = fallback_on_quota
        self.fallback_on_timeout = fallback_on_timeout
        self.fallback_on_5xx = fallback_on_5xx
        self.ollama_client = None

        if self.use_mock:
            self.mock_client = MockLLMClient()
        else:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Set use_mock=True for testing or provide the API key."
                )
            # Import and initialize google-genai SDK Client
            try:
                from google import genai

                self.genai_client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install it with: pip install google-genai"
                )

        # Initialize Ollama fallback if enabled
        if self.enable_fallback and ollama_enabled:
            self.ollama_client = OllamaClient(
                base_url=ollama_base_url,
                model=ollama_model,
                timeout=ollama_timeout,
            )
            logger.info(
                f"Ollama fallback configured: {ollama_base_url} ({ollama_model})"
            )

    def generate_stage1_response(self, prompt: str) -> str:

        if self.use_mock:
            return self.mock_client.generate_stage1_response(prompt)

        # Try primary provider (Gemini via SDK)
        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                temperature=0,
            )
            logger.debug("Stage 1 response generated via Gemini")
            return response.text
        except Exception as e:
            # Map google-genai SDK errors to our custom ProviderError types
            gemini_error = self._handle_gemini_error(e)
            
            if isinstance(gemini_error, ProviderQuotaError):
                if self.enable_fallback and self.fallback_on_quota:
                    logger.warning(
                        f"Gemini quota/credits exhausted; attempting Ollama fallback: {e}"
                    )
                    return self._try_ollama_fallback(prompt)
                raise gemini_error
            elif isinstance(gemini_error, ProviderTimeoutError):
                if self.enable_fallback and self.fallback_on_timeout:
                    logger.warning(
                        f"Gemini timeout; attempting Ollama fallback: {e}"
                    )
                    return self._try_ollama_fallback(prompt)
                raise gemini_error
            elif isinstance(gemini_error, ProviderServerError):
                if self.enable_fallback and self.fallback_on_5xx:
                    logger.warning(
                        f"Gemini server error; attempting Ollama fallback: {e}"
                    )
                    return self._try_ollama_fallback(prompt)
                raise gemini_error
            elif isinstance(gemini_error, ProviderUnavailableError):
                if self.enable_fallback:
                    logger.warning(
                        f"Gemini unavailable; attempting Ollama fallback: {e}"
                    )
                    return self._try_ollama_fallback(prompt)
                raise gemini_error
            else:
                # For other exceptions (JSON, validation, etc.), don't fallback
                # These are real bugs, not provider issues
                logger.error(f"Stage 1 generation failed: {type(e).__name__}: {e}")
                raise

    def _try_ollama_fallback(self, prompt: str) -> str:
        """
        Try Ollama as fallback provider.

        Args:
            prompt: Full prompt with context injected

        Returns:
            JSON string with extraction result

        Raises:
            ProviderError: If Ollama fallback fails
        """
        if not self.ollama_client:
            raise ProviderUnavailableError(
                "Ollama fallback requested but not configured. "
                "Set enable_fallback=True and ollama_enabled=True to use Ollama fallback."
            )

        try:
            logger.info(
                f"Attempting Stage 1 extraction with Ollama fallback "
                f"({self.ollama_client.model}) at {self.ollama_client.base_url}"
            )
            response = self.ollama_client.generate_stage1_response(prompt)
            logger.info("Stage 1 response generated via Ollama fallback")
            return response
        except ProviderError as e:
            logger.error(f"Ollama fallback failed: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during Ollama fallback: {type(e).__name__}: {e}"
            )
            raise ProviderUnavailableError(f"Ollama fallback failed: {e}")

    def _handle_gemini_error(self, error: Exception) -> ProviderError:
        """
        Map google-genai SDK errors to custom ProviderError types.

        Args:
            error: Exception from google-genai SDK

        Returns:
            ProviderError subclass matching the error type

        Raises:
            The original exception if it cannot be mapped
        """
        error_name = type(error).__name__
        error_message = str(error).lower()

        try:
            # Try to import google API error types
            from google.api_core import exceptions as google_exceptions
        except ImportError:
            # If google.api_core not available, fall back to string matching
            google_exceptions = None

        # Check for quota/rate limit errors (includes credit exhaustion)
        if google_exceptions and isinstance(error, google_exceptions.ResourceExhausted):
            logger.warning("Gemini ResourceExhausted: quota, rate limit, or credits exhausted")
            return ProviderQuotaError(
                "Gemini quota exhausted or credits depleted. Falling back to Ollama."
            )
        if "quota" in error_message or "rate_limit" in error_message:
            return ProviderQuotaError(f"Gemini quota error: {error}")
        if "resource" in error_message and "exhausted" in error_message:
            return ProviderQuotaError(f"Gemini resources exhausted: {error}")
        if "429" in error_message or "too many requests" in error_message:
            return ProviderQuotaError(f"Gemini rate limited (429): {error}")

        # Check for timeout errors
        if google_exceptions and isinstance(error, google_exceptions.DeadlineExceeded):
            logger.warning("Gemini DeadlineExceeded: request timed out")
            return ProviderTimeoutError("Gemini request timed out")
        if "deadline" in error_message or "timeout" in error_message:
            return ProviderTimeoutError(f"Gemini timeout: {error}")

        # Check for server/unavailable errors
        if google_exceptions and isinstance(
            error, google_exceptions.ServiceUnavailable
        ):
            logger.warning("Gemini ServiceUnavailable")
            return ProviderServerError("Gemini service temporarily unavailable")
        if google_exceptions and isinstance(error, google_exceptions.InternalServerError):
            logger.warning("Gemini InternalServerError")
            return ProviderServerError("Gemini internal server error")
        if "service" in error_message and "unavailable" in error_message:
            return ProviderServerError(f"Gemini unavailable: {error}")
        if "500" in error_message or "503" in error_message:
            return ProviderServerError(f"Gemini server error: {error}")

        # Check for connection/availability errors
        if "connection" in error_message or "unable to connect" in error_message:
            logger.warning(f"Gemini connection error: {error}")
            return ProviderUnavailableError(f"Cannot reach Gemini: {error}")
        if "ssl" in error_message or "certificate" in error_message:
            logger.warning(f"Gemini SSL/TLS error: {error}")
            return ProviderUnavailableError(f"Gemini SSL error: {error}")

        # For unmapped errors, return as a generic ProviderError for consistency
        logger.warning(f"Unmapped Gemini error: {error_name}: {error}")
        return ProviderUnavailableError(f"Gemini error: {error}")