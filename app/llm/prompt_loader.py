"""Prompt loading and caching."""
import logging
from pathlib import Path
from typing import Dict

from app.config import settings
from app.core.exceptions import PromptLoadException

logger = logging.getLogger(__name__)


class PromptLoader:
    """Load and cache extraction prompts."""

    def __init__(self):
        """Initialize the prompt loader with cache."""
        self._cache: Dict[str, str] = {}

    def load_extract_prompt(self, version: str) -> str:
        cache_key = f"extract_{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if version == "v1":
            path = settings.prompt_extract_v1_path
        elif version == "v2":
            path = settings.prompt_extract_v2_path
        else:
            raise PromptLoadException(f"Unknown extraction prompt version: {version}")

        try:
            prompt_path = Path(path)
            if not prompt_path.exists():
                raise PromptLoadException(
                    f"Prompt file not found: {prompt_path.absolute()}"
                )
            text = prompt_path.read_text(encoding="utf-8").strip()
            self._cache[cache_key] = text
            logger.debug(f"Loaded extract prompt {version} from {path}")
            return text
        except (IOError, OSError) as e:
            logger.error(f"Failed to load prompt {version}: {e}")
            raise PromptLoadException(f"Failed to load prompt {version}: {e}") from e

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()

    def load_interpret_prompt(self, version: str) -> str:
        """Load interpretation prompt by version.
        
        Args:
            version: Prompt version ("v1", etc.)
            
        Returns:
            Prompt text
            
        Raises:
            PromptLoadException: If prompt file not found
        """
        cache_key = f"interpret_{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if version == "v1":
            path = settings.prompt_interpret_v1_path
        else:
            raise PromptLoadException(f"Unknown interpretation prompt version: {version}")

        try:
            prompt_path = Path(path)
            if not prompt_path.exists():
                raise PromptLoadException(
                    f"Prompt file not found: {prompt_path.absolute()}"
                )
            text = prompt_path.read_text(encoding="utf-8").strip()
            self._cache[cache_key] = text
            logger.debug(f"Loaded interpret prompt {version} from {path}")
            return text
        except (IOError, OSError) as e:
            logger.error(f"Failed to load prompt {version}: {e}")
            raise PromptLoadException(f"Failed to load prompt {version}: {e}") from e


# Global loader instance
_loader = PromptLoader()


def load_extract_prompt(version: str) -> str:
    """Module-level function to load extraction prompt."""
    return _loader.load_extract_prompt(version)


def load_interpret_prompt(version: str) -> str:
    """Module-level function to load interpretation prompt."""
    return _loader.load_interpret_prompt(version)

