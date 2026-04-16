"""LLM-based prediction interpretation (Stage 2)."""
import json
import logging
from typing import Any, Dict

from app.llm.client import GeminiClient
from app.llm.prompt_loader import get_cached_interpretation_prompt
from app.models.feature_models import PropertyFeaturesComplete

logger = logging.getLogger(__name__)


class Stage2Interpreter:
    """
    Stage 2 prediction interpretation service.
    
    Takes a complete prediction (features + predicted price + global stats)
    and generates a human-friendly explanation of the result.
    
    Orchestrates the interpretation workflow:
    1. Load prompt template
    2. Inject context (features, predicted price, global stats)
    3. Call LLM
    4. Return interpretation text (plain text, no JSON)
    """

    def __init__(
        self,
        llm_client: GeminiClient,
        prompt_version: str = "v1",
    ):
        """
        Initialize the interpreter.
        
        Args:
            llm_client: LLM client instance
            prompt_version: Prompt version to use ("v1")
        """
        self.llm_client = llm_client
        self.prompt_version = prompt_version

    def interpret(
        self,
        complete_features: PropertyFeaturesComplete,
        predicted_price: float,
        global_stats: Dict[str, Any],
    ) -> str:
        """
        Generate an interpretation of the prediction.
        
        Args:
            complete_features: PropertyFeaturesComplete with all 12 required features
            predicted_price: Predicted sale price in USD
            global_stats: Global statistics from artifact (mean, median, min, max, quartiles)
        
        Returns:
            Plain text interpretation (2-3 sentences)
        
        Raises:
            ValueError: If LLM response is invalid or empty
            Exception: If prompt loading or LLM call fails
        """
        # Prepare feature context as readable JSON
        features_dict = complete_features.model_dump(exclude_none=False)
        features_json = json.dumps(features_dict, indent=2)

        # Extract stats for template (with defaults if missing)
        # Use real artifact keys: mean_price, median_price, min_price, max_price, q1_price, q3_price
        stats = {
            "mean": global_stats.get("mean_price", 0),
            "median": global_stats.get("median_price", 0),
            "min": global_stats.get("min_price", 0),
            "max": global_stats.get("max_price", 0),
            "q1": global_stats.get("q1_price", 0),
            "q3": global_stats.get("q3_price", 0),
        }

        # Load and prepare prompt
        prompt_template = get_cached_interpretation_prompt(self.prompt_version)

        # Inject context into template
        full_prompt = prompt_template.replace(
            "{{complete_features}}", features_json
        ).replace(
            "{{predicted_price}}", f"{predicted_price:,.2f}"
        ).replace(
            "{{global_stats_mean}}", f"{stats['mean']:,.2f}"
        ).replace(
            "{{global_stats_median}}", f"{stats['median']:,.2f}"
        ).replace(
            "{{global_stats_min}}", f"{stats['min']:,.2f}"
        ).replace(
            "{{global_stats_max}}", f"{stats['max']:,.2f}"
        ).replace(
            "{{global_stats_q1}}", f"{stats['q1']:,.2f}"
        ).replace(
            "{{global_stats_q3}}", f"{stats['q3']:,.2f}"
        )

        logger.debug(f"Calling LLM for interpretation with prompt version {self.prompt_version}")

        # Call LLM
        llm_response_text = self.llm_client.generate_stage1_response(full_prompt)

        # Validate response is not empty
        if not llm_response_text or not llm_response_text.strip():
            raise ValueError("LLM returned empty interpretation")

        logger.debug(f"Interpretation complete: {len(llm_response_text)} characters")

        return llm_response_text.strip()
