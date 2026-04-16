"""Tests for Ollama fallback when Gemini credits run out."""

import pytest
from unittest.mock import Mock, patch

from app.llm.client import (
    GeminiClient,
    ProviderQuotaError,
    ProviderTimeoutError,
    ProviderServerError,
    ProviderUnavailableError,
)
from app.config import settings


class TestOllamaFallback:
    """Test Gemini-to-Ollama fallback behavior."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a GeminiClient configured for fallback."""
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key-12345"},
            clear=False,
        ):
            with patch("google.genai.Client"):
                client = GeminiClient(
                    use_mock=False,
                    api_key="test-key-12345",
                    model_name="gemini-1.5-flash",
                    enable_fallback=True,
                    fallback_on_quota=True,
                    fallback_on_timeout=True,
                    fallback_on_5xx=True,
                    ollama_enabled=True,
                    ollama_base_url="http://localhost:11434",
                    ollama_model="llama3.2:1b",
                    ollama_timeout=30,
                )
                # Mock the genai client
                client.genai_client = Mock()
                yield client

    def test_quota_error_triggers_ollama_fallback(self, mock_gemini_client):
        """Test that Gemini quota error (credits exhausted) triggers Ollama fallback."""
        # Simulate Gemini returning a quota error
        gemini_error = Exception("429 Quota exceeded. Resource has been exhausted.")
        mock_gemini_client.genai_client.models.generate_content.side_effect = gemini_error

        # Mock Ollama client to return success
        mock_gemini_client.ollama_client = Mock()
        ollama_response = '{"extracted_features": {"OverallQual": 8}}'
        mock_gemini_client.ollama_client.generate_stage1_response.return_value = (
            ollama_response
        )

        prompt = "Extract: 8-bedroom house"
        response = mock_gemini_client.generate_stage1_response(prompt)

        # Verify Ollama was called as fallback
        mock_gemini_client.ollama_client.generate_stage1_response.assert_called_once_with(
            prompt
        )
        assert response == ollama_response

    def test_resource_exhausted_exception_maps_to_quota_error(self, mock_gemini_client):
        """Test that google.api_core ResourceExhausted error is properly mapped."""
        try:
            from google.api_core.exceptions import ResourceExhausted
        except ImportError:
            pytest.skip("google.api_core not installed")

        # Simulate ResourceExhausted from google-genai
        gemini_error = ResourceExhausted("Quota exceeded")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        # Mock Ollama to succeed
        mock_gemini_client.ollama_client = Mock()
        mock_gemini_client.ollama_client.generate_stage1_response.return_value = (
            '{"extracted_features": {}}'
        )

        # Should fallback and not raise
        response = mock_gemini_client.generate_stage1_response("test")
        assert response == '{"extracted_features": {}}'

    def test_quota_error_raised_if_fallback_disabled(self, mock_gemini_client):
        """Test that quota error is raised if fallback is disabled."""
        mock_gemini_client.enable_fallback = False
        gemini_error = Exception("429 Quota exceeded")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        prompt = "Extract: property details"

        # Should raise ProviderQuotaError since fallback is disabled
        with pytest.raises(ProviderQuotaError):
            mock_gemini_client.generate_stage1_response(prompt)

    def test_timeout_error_triggers_ollama_fallback(self, mock_gemini_client):
        """Test that Gemini timeout triggers Ollama fallback."""
        gemini_error = Exception("Request deadline exceeded")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        # Mock Ollama client
        mock_gemini_client.ollama_client = Mock()
        mock_gemini_client.ollama_client.generate_stage1_response.return_value = (
            '{"timeout": false}'
        )

        response = mock_gemini_client.generate_stage1_response("test")
        mock_gemini_client.ollama_client.generate_stage1_response.assert_called_once()

    def test_server_error_triggers_ollama_fallback(self, mock_gemini_client):
        """Test that Gemini 5xx errors trigger Ollama fallback."""
        gemini_error = Exception("500 Internal Server Error")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        # Mock Ollama client
        mock_gemini_client.ollama_client = Mock()
        mock_gemini_client.ollama_client.generate_stage1_response.return_value = (
            '{"status": "ok"}'
        )

        response = mock_gemini_client.generate_stage1_response("test")
        mock_gemini_client.ollama_client.generate_stage1_response.assert_called_once()

    def test_ollama_fallback_fails_raises_error(self, mock_gemini_client):
        """Test that if both Gemini and Ollama fail, error is raised."""
        gemini_error = Exception("429 Quota exceeded")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        # Ollama client throws error too
        mock_gemini_client.ollama_client = Mock()
        mock_gemini_client.ollama_client.generate_stage1_response.side_effect = (
            ProviderUnavailableError("Ollama unreachable")
        )

        # Should raise since both providers failed
        with pytest.raises(ProviderUnavailableError):
            mock_gemini_client.generate_stage1_response("test")

    def test_ollama_not_configured_raises_error(self, mock_gemini_client):
        """Test that fallback fails gracefully if Ollama not configured."""
        mock_gemini_client.ollama_client = None
        gemini_error = Exception("429 Quota exceeded")
        mock_gemini_client.genai_client.models.generate_content.side_effect = (
            gemini_error
        )

        # Should raise error about missing Ollama config
        with pytest.raises(ProviderUnavailableError) as exc_info:
            mock_gemini_client.generate_stage1_response("test")

        assert "not configured" in str(exc_info.value)

    def test_error_mapping_detects_quota_keywords(self, mock_gemini_client):
        """Test that quota error detection catches common keywords."""
        test_cases = [
            "quota exceeded",
            "rate_limit reached",
            "resource exhausted",
            "429 Too Many Requests",
        ]

        for error_message in test_cases:
            error = Exception(error_message)
            provider_error = mock_gemini_client._handle_gemini_error(error)
            assert isinstance(
                provider_error, ProviderQuotaError
            ), f"Failed for: {error_message}"

    def test_error_mapping_detects_timeout_keywords(self, mock_gemini_client):
        """Test that timeout error detection catches common keywords."""
        test_cases = [
            "deadline exceeded",
            "request timeout",
            "timeout waiting for response",
        ]

        for error_message in test_cases:
            error = Exception(error_message)
            provider_error = mock_gemini_client._handle_gemini_error(error)
            assert isinstance(
                provider_error, ProviderTimeoutError
            ), f"Failed for: {error_message}"

    def test_error_mapping_detects_server_errors(self, mock_gemini_client):
        """Test that server error detection catches common patterns."""
        test_cases = [
            "500 internal server error",
            "503 service unavailable",
            "service unavailable temporarily",
        ]

        for error_message in test_cases:
            error = Exception(error_message)
            provider_error = mock_gemini_client._handle_gemini_error(error)
            assert isinstance(
                provider_error, ProviderServerError
            ), f"Failed for: {error_message}"

    def test_error_mapping_detects_connection_errors(self, mock_gemini_client):
        """Test that connection error detection works."""
        test_cases = [
            "connection refused",
            "unable to connect to server",
            "ssl certificate error",
        ]

        for error_message in test_cases:
            error = Exception(error_message)
            provider_error = mock_gemini_client._handle_gemini_error(error)
            assert isinstance(
                provider_error, ProviderUnavailableError
            ), f"Failed for: {error_message}"
