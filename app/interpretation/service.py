"""Stage 2 interpretation service for price predictions."""
import json
import logging
from typing import Any, Dict, Optional

from app.core.exceptions import LLMException
from app.llm.openrouter_client import OpenRouterClient
from app.llm.prompt_loader import load_interpret_prompt

logger = logging.getLogger(__name__)


class InterpretationService:
    """Orchestrate Stage 2 interpretation of price predictions."""

    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        prompt_version: str = "v1",
    ):
        """Initialize the interpretation service.
        
        Args:
            client: OpenRouter client (if None, creates default)
            prompt_version: Interpretation prompt version ("v1", etc.)
        """
        self.client = client or OpenRouterClient()
        self.prompt_version = prompt_version

    def interpret(
        self,
        complete_features: Dict[str, Any],
        predicted_price: float,
        global_stats: Dict[str, Any],
    ) -> str:
        """Interpret a price prediction using LLM.
        
        Args:
            complete_features: Dict of all 12 required features with final values
            predicted_price: Predicted sale price from the model
            global_stats: Dict with median_price, mean_price, min_price, max_price,
                         q1_price, q3_price, typical_range
            
        Returns:
            Plain text interpretation of the prediction
            
        Raises:
            LLMException: If interpretation fails
        """
        # Load and format the interpretation prompt
        system_prompt = load_interpret_prompt(self.prompt_version)

        # Build the user context with feature values and statistics
        user_context = self._build_user_context(
            complete_features, predicted_price, global_stats
        )

        logger.debug(
            f"Calling interpretation with prompt version {self.prompt_version}"
        )

        # Call LLM for plain text response
        try:
            interpretation = self.client.call_plain_text(
                system_prompt, user_context
            )
        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            raise LLMException(f"Interpretation failed: {e}") from e

        # Validate response
        if not interpretation or not interpretation.strip():
            logger.error("Interpretation returned empty response")
            raise LLMException("Interpretation returned empty response")

        logger.info(f"Interpretation complete: {len(interpretation)} chars")
        return interpretation.strip()

    def _build_user_context(
        self,
        complete_features: Dict[str, Any],
        predicted_price: float,
        global_stats: Dict[str, Any],
    ) -> str:
        """Build the user message context for the LLM.
        
        Args:
            complete_features: Dict of all 12 required features
            predicted_price: Predicted price
            global_stats: Dataset statistics
            
        Returns:
            Formatted user message for LLM
        """
        # Format features as a readable string
        features_str = "\n  ".join(
            f"{k}: {v}" for k, v in complete_features.items()
        )

        # Extract statistics safely and format them
        def format_stat(key: str, value_or_none: Any) -> str:
            """Format a statistic value for display."""
            if value_or_none is None or value_or_none == "N/A":
                return "N/A"
            try:
                return f"${float(value_or_none):,.0f}"
            except (TypeError, ValueError):
                return str(value_or_none)

        median_str = format_stat("median", global_stats.get("median_price"))
        mean_str = format_stat("mean", global_stats.get("mean_price"))
        min_str = format_stat("min", global_stats.get("min_price"))
        max_str = format_stat("max", global_stats.get("max_price"))
        q1_str = format_stat("q1", global_stats.get("q1_price"))
        q3_str = format_stat("q3", global_stats.get("q3_price"))

        # Build context
        context = f"""
Features:
  {features_str}

Predicted Price: ${predicted_price:,.2f}

Dataset Statistics:
  Median: {median_str}
  Mean: {mean_str}
  Min: {min_str}
  Max: {max_str}
  Q1 (25th percentile): {q1_str}
  Q3 (75th percentile): {q3_str}
  Typical Range (Q1-Q3): {q1_str}-{q3_str}
"""
        return context.strip()
