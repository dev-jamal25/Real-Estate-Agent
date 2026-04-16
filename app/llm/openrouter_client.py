"""OpenRouter client for calling the LLM API directly."""
import json
import logging
from typing import Any, Dict

import requests

from app.config import settings
from app.core.exceptions import LLMException

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Direct HTTP client for OpenRouter API.
    
    Attributes:
        api_key: API key from environment
        base_url: API endpoint base URL
        model: Model identifier
        temperature: Sampling temperature
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0,
    ):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            model: Model name (defaults to settings)
            temperature: Sampling temperature
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.openrouter_model
        self.temperature = temperature

    def call(
        self, system_prompt: str, user_message: str
    ) -> Dict[str, Any]:
        """Call the OpenRouter API with structured JSON response.
        
        Args:
            system_prompt: System instruction prompt
            user_message: User's message
            
        Returns:
            Parsed JSON response from the model
            
        Raises:
            LLMException: If API call fails or response is invalid
        """
        if not self.api_key:
            raise LLMException(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY env var."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise LLMException(f"OpenRouter API error: {e}") from e

        try:
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            raise LLMException(f"Invalid response format: {e}") from e

    def call_plain_text(
        self, system_prompt: str, user_message: str
    ) -> str:
        """Call the OpenRouter API expecting plain text response.
        
        Args:
            system_prompt: System instruction prompt
            user_message: User's message
            
        Returns:
            Plain text response from the model
            
        Raises:
            LLMException: If API call fails or response is invalid
        """
        if not self.api_key:
            raise LLMException(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY env var."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise LLMException(f"OpenRouter API error: {e}") from e

        try:
            response_json = response.json()
        except Exception as e:
            logger.error(f"Failed to parse OpenRouter response as JSON: {e}")
            raise LLMException(f"Invalid JSON response: {e}") from e

        try:
            content = response_json["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise LLMException("Empty response from LLM")
            return content.strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract text from OpenRouter response: {e}")
            raise LLMException(f"Invalid response format: {e}") from e
