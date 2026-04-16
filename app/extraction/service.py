"""Main extraction service orchestrating the Stage 1 pipeline."""
import json
import logging
from typing import Any, Dict, Optional

from app.contracts.property_features import REQUIRED_FEATURES
from app.core.exceptions import ExtractionException
from app.extraction.normalizer import FeatureNormalizer
from app.extraction.schemas import ExtractedFeatures, LLMExtractionOutput, Stage1Response
from app.llm.openrouter_client import OpenRouterClient
from app.llm.prompt_loader import load_extract_prompt

logger = logging.getLogger(__name__)


class ExtractionService:
    """Orchestrate the Stage 1 extraction pipeline."""

    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        prompt_version: str = "v1",
    ):
        """Initialize the extraction service.
        
        Args:
            client: OpenRouter client (if None, creates default)
            prompt_version: Extraction prompt version ("v1" or "v2")
        """
        self.client = client or OpenRouterClient()
        self.prompt_version = prompt_version

    def extract(
        self,
        latest_message: str,
        accumulated_state: Optional[Dict[str, Any]] = None,
    ) -> Stage1Response:
        """Extract features from user message and accumulated state.
        
        Args:
            latest_message: Latest user input message
            accumulated_state: Dict of previously extracted features (may include None values)
            
        Returns:
            Stage1Response with extracted features, missing list, completeness, and reply
            
        Raises:
            ExtractionException: If extraction or normalization fails
        """
        if accumulated_state is None:
            accumulated_state = {}

        # Load and format the extraction prompt
        system_prompt = load_extract_prompt(self.prompt_version)

        # Build the user message context
        user_context = self._build_user_context(
            latest_message, accumulated_state
        )

        logger.debug(
            f"Calling OpenRouter with prompt version {self.prompt_version}"
        )

        # Call LLM
        try:
            raw_response = self.client.call(system_prompt, user_context)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise ExtractionException(f"LLM call failed: {e}") from e

        # Parse the response
        try:
            llm_output = LLMExtractionOutput(**raw_response)
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            raise ExtractionException(f"Invalid LLM response structure: {e}") from e

        # Normalize extracted features
        normalized = FeatureNormalizer.normalize_all(llm_output.extracted_features)

        # Merge with accumulated state (normalized takes precedence)
        merged_state = {**accumulated_state, **normalized}

        # Determine what is still missing
        missing_features = self._compute_missing_features(merged_state)
        is_complete = len(missing_features) == 0

        logger.info(
            f"Extraction complete: {len(merged_state) - len([v for v in merged_state.values() if v is None])}/{len(REQUIRED_FEATURES)} features, missing={missing_features}"
        )

        # Build the final response (includes accumulated state + this turn's extraction)
        extracted_features_obj = ExtractedFeatures(**merged_state)

        return Stage1Response(
            extracted_features=extracted_features_obj,
            missing_features=missing_features,
            is_complete=is_complete,
            reply=llm_output.reply,
        )

    @staticmethod
    def _build_user_context(
        latest_message: str, accumulated_state: Dict[str, Any]
    ) -> str:
        """Build the context string for the LLM.
        
        Args:
            latest_message: Latest user input
            accumulated_state: Accumulated feature state
            
        Returns:
            Formatted context for the extraction prompt
        """
        # Summarize what we already know
        known_features = {
            k: v for k, v in accumulated_state.items() if v is not None
        }
        known_str = json.dumps(known_features, indent=2) if known_features else "None"

        context = f"""
Latest user message:
{latest_message}

Previously extracted features:
{known_str}

Please extract any new features from the latest message and return the result in JSON format.
""".strip()
        return context

    @staticmethod
    def _compute_missing_features(state: Dict[str, Any]) -> list[str]:
        """Compute which required features are still missing.
        
        Args:
            state: Current feature state
            
        Returns:
            List of missing feature names
        """
        missing = []
        for feature in REQUIRED_FEATURES:
            value = state.get(feature)
            if value is None:
                missing.append(feature)
        return missing
