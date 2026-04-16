"""LLM-based feature extraction (Stage 1)."""
import json
import logging
from typing import Optional

from app.core.validators import (
    build_follow_up_message,
    compute_missing_features,
    is_complete,
    normalize_feature_value,
)
from app.llm.client import GeminiClient
from app.llm.prompt_loader import get_cached_stage1_prompt
from app.models.feature_models import PropertyFeaturesPartial
from app.models.response_models import Stage1ExtractionResult

logger = logging.getLogger(__name__)


class Stage1Extractor:
    """
    Stage 1 feature extraction service.
    
    Orchestrates the extraction workflow:
    1. Load prompt template
    2. Inject context (user message, current state)
    3. Call LLM
    4. Parse and validate JSON response
    5. Normalize extracted values
    6. Compute missing features
    7. Return structured result
    """

    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_version: str = "v2",
    ):
        self.llm_client = llm_client or GeminiClient(use_mock=True)
        self.prompt_version = prompt_version

    def extract(
        self,
        latest_user_message: str,
        current_known_feature_state: Optional[PropertyFeaturesPartial] = None,
    ) -> Stage1ExtractionResult:
        # Prepare context
        feature_overrides_dict = (
            current_known_feature_state.model_dump(exclude_none=True)
            if current_known_feature_state
            else {}
        )

        # Load and prepare prompt
        prompt_template = get_cached_stage1_prompt(self.prompt_version)
        
        # Inject context into template
        full_prompt = prompt_template.replace(
            "{{latest_user_message}}", latest_user_message
        ).replace("{{feature_overrides}}", json.dumps(feature_overrides_dict, indent=2))

        logger.debug(f"Calling LLM with prompt version {self.prompt_version}")

        # Call LLM
        llm_response_text = self.llm_client.generate_stage1_response(full_prompt)

        # Parse JSON response
        try:
            response_json = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {llm_response_text}")
            raise ValueError(
                f"LLM returned invalid JSON: {str(e)}"
            ) from e

        # Validate response structure
        required_keys = {"extracted_features", "missing_features", "is_complete", "assistant_message"}
        if not required_keys.issubset(response_json.keys()):
            raise ValueError(
                f"LLM response missing required keys. Expected {required_keys}, "
                f"got {set(response_json.keys())}"
            )

        # Normalize extracted features
        raw_extracted = response_json.get("extracted_features", {})
        normalized_extracted = self._normalize_extracted_features(raw_extracted)

        # Compute actual missing features using accumulated state
        accumulated_state = {
            **feature_overrides_dict,
            **normalized_extracted,
        }
        computed_missing = compute_missing_features(
            normalized_extracted, feature_overrides_dict
        )
        computed_is_complete = is_complete(normalized_extracted, feature_overrides_dict)

        # CODE OWNS FOLLOW-UP MESSAGE: Generate assistant message based on completeness
        # Do not use LLM's assistant_message; it will be discarded
        if computed_is_complete:
            assistant_message = "All required features collected. Ready to predict."
        else:
            assistant_message = build_follow_up_message(computed_missing)

        # Construct result
        result = Stage1ExtractionResult(
            extracted_features=PropertyFeaturesPartial(**normalized_extracted),
            missing_features=computed_missing,
            is_complete=computed_is_complete,
            assistant_message=assistant_message,
        )

        logger.debug(
            f"Extraction complete: {len(normalized_extracted)} features extracted, "
            f"{len(computed_missing)} missing, complete={computed_is_complete}"
        )

        return result

    def _normalize_extracted_features(self, raw_features: dict) -> dict:
        normalized = {}

        for feature_name, raw_value in raw_features.items():
            # Skip unknown feature names
            normalized_value = normalize_feature_value(feature_name, raw_value)
            
            if normalized_value is not None:
                normalized[feature_name] = normalized_value
            else:
                logger.debug(
                    f"Skipping invalid feature value: {feature_name}={raw_value}"
                )

        return normalized

