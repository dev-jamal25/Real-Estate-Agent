"""Prompt loading and management."""
from pathlib import Path
from typing import Dict

# Prompts are stored in prompts/ folder at the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(filename: str) -> str:
    """
    Load a prompt from the prompts folder.
    
    Args:
        filename: Prompt filename (e.g., "extract_v1.txt")
    
    Returns:
        Prompt text content
    
    Raises:
        FileNotFoundError: If the prompt file does not exist
    """
    prompt_path = PROMPTS_DIR / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            f"Make sure {filename} exists in {PROMPTS_DIR}"
        )
    
    with open(prompt_path, "r") as f:
        return f.read()


def load_stage1_extraction_prompt(version: str = "v1") -> str:
    """
    Load the Stage 1 extraction prompt.
    
    Args:
        version: Prompt version ("v1" or "v2")
    
    Returns:
        Prompt text with template variables ready for injection
    
    Raises:
        ValueError: If version is not supported
        FileNotFoundError: If the prompt file doesn't exist
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported prompt version: {version}. Use 'v1' or 'v2'.")
    
    # Map version to actual filename: v1 -> extract_v1.txt
    filename = f"extract_{version}.txt"
    return load_prompt(filename)


_prompt_cache: Dict[str, str] = {}


def get_cached_stage1_prompt(version: str = "v1") -> str:
    """
    Get Stage 1 prompt with caching to avoid repeated file I/O.
    
    Args:
        version: Prompt version ("v1" or "v2")
    
    Returns:
        Cached prompt text
    """
    cache_key = f"stage1_{version}"
    
    if cache_key not in _prompt_cache:
        _prompt_cache[cache_key] = load_stage1_extraction_prompt(version)
    
    return _prompt_cache[cache_key]


def load_interpretation_prompt(version: str = "v1") -> str:
    """
    Load the Stage 2 interpretation prompt.
    
    Args:
        version: Prompt version ("v1")
    
    Returns:
        Prompt text with template variables ready for injection
    
    Raises:
        ValueError: If version is not supported
        FileNotFoundError: If the prompt file doesn't exist
    """
    if version not in ("v1",):
        raise ValueError(f"Unsupported prompt version: {version}. Use 'v1'.")
    
    # Map version to actual filename: v1 -> interpret_v1.txt
    filename = f"interpret_{version}.txt"
    return load_prompt(filename)


def get_cached_interpretation_prompt(version: str = "v1") -> str:
    """
    Get Stage 2 interpretation prompt with caching to avoid repeated file I/O.
    
    Args:
        version: Prompt version ("v1")
    
    Returns:
        Cached prompt text
    """
    cache_key = f"interpret_{version}"
    
    if cache_key not in _prompt_cache:
        _prompt_cache[cache_key] = load_interpretation_prompt(version)
    
    return _prompt_cache[cache_key]

