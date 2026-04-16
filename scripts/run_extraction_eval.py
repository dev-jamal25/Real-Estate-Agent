#!/usr/bin/env python
"""Run comprehensive evaluation on Stage 1 extraction prompts.

Tests both v1 and v2 prompt versions across all evaluation cases.
Logs version, input, output, validation result to JSONL.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.constants import EVAL_LOG_FILE
from app.extraction.normalizer import FeatureNormalizer
from app.extraction.schemas import ExtractedFeatures, Stage1Response
from app.extraction.service import ExtractionService
from app.llm.openrouter_client import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExtractionPromptEvaluator:
    """Evaluate Stage 1 extraction prompts comprehensively."""

    def __init__(
        self,
        eval_cases_path: str = "evals/extraction_eval_cases.yaml",
        output_path: str = None,
        use_mock: bool = False,
    ):
        """Initialize evaluator.
        
        Args:
            eval_cases_path: Path to YAML evaluation cases
            output_path: Path to write JSONL results (default from constants)
            use_mock: If True, use mock responses instead of real API calls
        """
        self.eval_cases_path = eval_cases_path
        self.output_path = output_path or EVAL_LOG_FILE
        self.use_mock = use_mock
        self.results = []

    def load_cases(self) -> list:
        """Load evaluation test cases from YAML."""
        try:
            with open(self.eval_cases_path, "r") as f:
                data = yaml.safe_load(f)

            # Extract cases list (skip meta section)
            cases = data.get("cases", [])
            logger.info(f"Loaded {len(cases)} evaluation cases from {self.eval_cases_path}")
            return cases
        except Exception as e:
            logger.error(f"Failed to load eval cases: {e}")
            raise

    def evaluate_case(
        self,
        case: Dict[str, Any],
        version: str,
    ) -> Dict[str, Any]:
        """Evaluate a single case with a specific prompt version.
        
        Args:
            case: Test case dict with id, description, input_query, current_known_feature_state, expected_json
            version: Prompt version ("v1" or "v2")
            
        Returns:
            Dict with version, input, output, validation result, timestamp
        """
        case_id = case.get("id", "unknown")
        input_query = case.get("input_query", "")
        current_state = case.get("current_known_feature_state", {})
        expected = case.get("expected_json", {})

        logger.info(f"Evaluating {case_id} with version {version}")

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "version": version,
            "input": {
                "query": input_query,
                "current_known_features": current_state,
            },
            "output": None,
            "expected": expected,
            "validation": {
                "passed": False,
                "errors": [],
            },
        }

        try:
            # Run extraction
            client = OpenRouterClient()
            service = ExtractionService(client=client, prompt_version=version)
            response = service.extract(input_query, accumulated_state=current_state)

            # Capture output
            result["output"] = {
                "extracted_features": response.extracted_features.model_dump(),
                "missing_features": response.missing_features,
                "is_complete": response.is_complete,
                "reply": response.reply[:200],  # Truncate long replies
            }

            # Validate against expected
            validation_errors = self._validate_response(response, expected)
            result["validation"]["errors"] = validation_errors
            result["validation"]["passed"] = len(validation_errors) == 0

            logger.info(
                f"  {case_id}/{version}: {'✓ PASS' if result['validation']['passed'] else '✗ FAIL'}"
            )
            if validation_errors:
                for error in validation_errors:
                    logger.warning(f"    - {error}")

        except Exception as e:
            logger.error(f"  {case_id}/{version}: ERROR - {e}")
            result["output"] = None
            result["validation"]["passed"] = False
            result["validation"]["errors"] = [str(e)]

        return result

    def _validate_response(
        self, response: Stage1Response, expected: Dict[str, Any]
    ) -> list:
        """Validate response against expected output.
        
        Args:
            response: Extraction service response
            expected: Expected output dict with extracted_features, missing_features, is_complete, reply
            
        Returns:
            List of validation error messages (empty if all pass)
        """
        errors = []

        # Check is_complete
        expected_complete = expected.get("is_complete", False)
        if response.is_complete != expected_complete:
            errors.append(
                f"is_complete mismatch: got {response.is_complete}, expected {expected_complete}"
            )

        # Check extracted_features keys
        expected_features = expected.get("extracted_features", {})
        actual_keys = set(
            k for k, v in response.extracted_features.model_dump().items() if v is not None
        )
        expected_keys = set(expected_features.keys())

        missing_keys = expected_keys - actual_keys
        if missing_keys:
            errors.append(f"Missing extracted features: {missing_keys}")

        extra_keys = actual_keys - expected_keys
        if extra_keys:
            errors.append(f"Unexpected extracted features: {extra_keys}")

        # Check extracted_features values
        for key in expected_keys & actual_keys:
            expected_val = expected_features[key]
            actual_val = getattr(response.extracted_features, key)
            if actual_val != expected_val:
                errors.append(
                    f"Feature {key} value mismatch: got {actual_val}, expected {expected_val}"
                )

        # Check missing_features
        expected_missing = set(expected.get("missing_features", []))
        actual_missing = set(response.missing_features)

        if actual_missing != expected_missing:
            errors.append(
                f"missing_features mismatch: got {actual_missing}, expected {expected_missing}"
            )

        return errors

    def run_all(self, versions: list = None) -> None:
        """Run evaluation on all cases for specified versions.
        
        Args:
            versions: List of versions to test (default ["v1", "v2"])
        """
        if versions is None:
            versions = ["v1", "v2"]

        cases = self.load_cases()
        total_cases = len(cases) * len(versions)

        logger.info(f"Starting evaluation: {len(cases)} cases × {len(versions)} versions = {total_cases} tests")

        for case in cases:
            for version in versions:
                result = self.evaluate_case(case, version)
                self.results.append(result)

        logger.info(f"Evaluation complete. Saving {len(self.results)} results...")
        self.save_results()

    def save_results(self) -> None:
        """Save evaluation results to JSONL."""
        try:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for result in self.results:
                    f.write(json.dumps(result) + "\n")

            logger.info(f"Saved {len(self.results)} results to {output_path}")

            # Print summary
            self._print_summary()
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def _print_summary(self) -> None:
        """Print evaluation summary."""
        passed = sum(1 for r in self.results if r["validation"]["passed"])
        failed = len(self.results) - passed

        by_version = {}
        for result in self.results:
            version = result["version"]
            if version not in by_version:
                by_version[version] = {"passed": 0, "failed": 0}
            if result["validation"]["passed"]:
                by_version[version]["passed"] += 1
            else:
                by_version[version]["failed"] += 1

        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total: {passed} passed, {failed} failed out of {len(self.results)} tests")
        for version in sorted(by_version.keys()):
            stats = by_version[version]
            logger.info(
                f"  {version}: {stats['passed']} passed, {stats['failed']} failed "
                f"({100 * stats['passed'] / (stats['passed'] + stats['failed']):.0f}%)"
            )
        logger.info("=" * 60)


def main():
    """Main entry point."""
    # Create evaluator
    evaluator = ExtractionPromptEvaluator(
        eval_cases_path="evals/extraction_eval_cases.yaml",
        output_path="logs/extraction_prompt_eval.jsonl",
        use_mock=False,
    )

    # Run evaluation on both versions
    try:
        evaluator.run_all(versions=["v1", "v2"])
        logger.info("Evaluation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
