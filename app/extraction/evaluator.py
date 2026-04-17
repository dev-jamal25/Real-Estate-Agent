"""Prompt evaluation and comparison."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.core.constants import EVAL_LOG_FILE
from app.core.exceptions import ExtractionException
from app.extraction.service import ExtractionService
from app.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class ExtractionEvaluator:
    """Evaluate extraction prompts on test cases."""

    def __init__(self, eval_cases_path: str = "evals/extraction_eval_cases.yaml"):
        """Initialize the evaluator.
        
        Args:
            eval_cases_path: Path to YAML file with test cases
        """
        self.eval_cases_path = eval_cases_path
        self.eval_results: List[Dict[str, Any]] = []

    def load_eval_cases(self) -> List[Dict[str, Any]]:
        """Load evaluation test cases from YAML.
        
        Returns:
            List of test case dicts with 'name' and 'message' keys
            
        Raises:
            ExtractionException: If file not found or malformed
        """
        try:
            path = Path(self.eval_cases_path)
            if not path.exists():
                raise ExtractionException(
                    f"Eval cases file not found: {path.absolute()}"
                )
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, list):
                raise ExtractionException("Eval cases file must contain a YAML list")
            return data
        except (yaml.YAMLError, IOError) as e:
            raise ExtractionException(f"Failed to load eval cases: {e}") from e

    def evaluate(
        self, versions: List[str] = None, mock_responses: Dict[str, Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if versions is None:
            versions = ["v1", "v2"]

        eval_cases = self.load_eval_cases()
        logger.info(f"Loaded {len(eval_cases)} evaluation cases")

        for case in eval_cases:
            case_name = case.get("name", "unnamed")
            user_message = case.get("message", "")

            logger.info(f"Evaluating case: {case_name}")

            case_result = {"case_name": case_name, "results": {}}

            for version in versions:
                logger.debug(f"Running prompt version {version}")

                try:
                    result = self._evaluate_single(
                        version, user_message, mock_responses
                    )
                    case_result["results"][version] = result
                except Exception as e:
                    logger.error(
                        f"Evaluation failed for {case_name}/{version}: {e}"
                    )
                    case_result["results"][version] = {
                        "error": str(e),
                        "validation_passed": False,
                    }

            self.eval_results.append(case_result)

        return self.eval_results

    def _evaluate_single(
        self,
        version: str,
        user_message: str,
        mock_responses: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Create client (use mock if provided)
        if mock_responses and version in mock_responses:
            client = MockOpenRouterClient(mock_responses[version])
        else:
            client = OpenRouterClient()

        # Create service with the specified version
        service = ExtractionService(client=client, prompt_version=version)

        # Run extraction
        response = service.extract(user_message, accumulated_state={})

        # Log the result
        result = {
            "output": {
                "extracted_features": response.extracted_features.model_dump(),
                "missing_features": response.missing_features,
                "is_complete": response.is_complete,
                "reply": response.reply[:100],  # Truncate long replies
            },
            "validation_passed": True,  # If we got here, validation passed
        }

        return result

    def save_results(self, output_path: str = None) -> None:
        """Save evaluation results to JSONL.
        
        Args:
            output_path: Path to write JSONL (default: from constants)
        """
        if output_path is None:
            output_path = EVAL_LOG_FILE

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                for result in self.eval_results:
                    f.write(json.dumps(result) + "\n")

            logger.info(f"Saved {len(self.eval_results)} evaluation results to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise ExtractionException(f"Failed to save results: {e}") from e


class MockOpenRouterClient:
    """Mock OpenRouter client for testing."""

    def __init__(self, mock_response: Dict[str, Any]):
        """Initialize with a mock response.
        
        Args:
            mock_response: The JSON response to return
        """
        self.mock_response = mock_response

    def call(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        """Return the mock response."""
        return self.mock_response
