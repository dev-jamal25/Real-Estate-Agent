"""Tests for the extraction evaluator."""
import json
from pathlib import Path

import pytest

from app.extraction.evaluator import ExtractionEvaluator


class TestExtractionEvaluator:
    """Test the extraction evaluator."""

    def test_load_eval_cases(self, tmp_path):
        """Should load eval cases from YAML."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Test 1"
  message: "User message 1"
- name: "Test 2"
  message: "User message 2"
"""
        )

        evaluator = ExtractionEvaluator(str(eval_file))
        cases = evaluator.load_eval_cases()

        assert len(cases) == 2
        assert cases[0]["name"] == "Test 1"
        assert cases[1]["message"] == "User message 2"

    def test_load_eval_cases_not_found(self):
        """Should raise exception if file not found."""
        from app.core.exceptions import ExtractionException

        evaluator = ExtractionEvaluator("nonexistent.yaml")
        with pytest.raises(ExtractionException):
            evaluator.load_eval_cases()

    def test_load_eval_cases_invalid_yaml(self, tmp_path):
        """Should raise exception on invalid YAML."""
        from app.core.exceptions import ExtractionException

        eval_file = tmp_path / "invalid.yaml"
        eval_file.write_text("{ invalid yaml ][")

        evaluator = ExtractionEvaluator(str(eval_file))
        with pytest.raises(ExtractionException):
            evaluator.load_eval_cases()

    def test_evaluate_with_mock_responses(self, tmp_path):
        """Should evaluate with mock responses."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Case 1"
  message: "3000 sq ft, 2 baths"
"""
        )

        mock_responses = {
            "v1": {
                "extracted_features": {"GrLivArea": 3000, "FullBath": 2},
                "missing_features": ["OverallQual"],
                "reply": "Got it",
            },
            "v2": {
                "extracted_features": {"GrLivArea": 3000, "FullBath": 2},
                "missing_features": ["OverallQual"],
                "reply": "Understood",
            },
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        results = evaluator.evaluate(
            versions=["v1", "v2"], mock_responses=mock_responses
        )

        assert len(results) == 1
        assert results[0]["case_name"] == "Case 1"
        assert "v1" in results[0]["results"]
        assert "v2" in results[0]["results"]
        assert results[0]["results"]["v1"]["validation_passed"] is True
        assert results[0]["results"]["v2"]["validation_passed"] is True

    def test_save_results(self, tmp_path):
        """Should save results to JSONL."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Test case"
  message: "Test message"
"""
        )

        output_file = tmp_path / "results.jsonl"

        mock_responses = {
            "v1": {
                "extracted_features": {"FullBath": 2},
                "missing_features": ["OverallQual"],
                "reply": "OK",
            }
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        evaluator.evaluate(versions=["v1"], mock_responses=mock_responses)
        evaluator.save_results(str(output_file))

        assert output_file.exists()

        # Read and verify JSONL
        with open(output_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["case_name"] == "Test case"
            assert "results" in data

    def test_save_results_creates_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Test"
  message: "Message"
"""
        )

        output_dir = tmp_path / "subdir" / "results.jsonl"

        mock_responses = {
            "v1": {
                "extracted_features": {},
                "missing_features": [],
                "reply": "OK",
            }
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        evaluator.evaluate(versions=["v1"], mock_responses=mock_responses)
        evaluator.save_results(str(output_dir))

        assert output_dir.exists()

    def test_evaluate_handles_errors(self, tmp_path):
        """Should handle errors in individual evaluations."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Case 1"
  message: "Test"
- name: "Case 2"
  message: "Test 2"
"""
        )

        # Only provide response for Case 1 (Case 2 will fail to normalize)
        mock_responses = {
            "v1": {
                "extracted_features": {"GrLivArea": 3000},
                "missing_features": [],
                "reply": "OK",
            }
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        results = evaluator.evaluate(versions=["v1"], mock_responses=mock_responses)

        # Should have results for both cases despite error
        assert len(results) == 2

    def test_multiple_eval_cases(self, tmp_path):
        """Should evaluate multiple cases."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Case 1"
  message: "Message 1"
- name: "Case 2"
  message: "Message 2"
- name: "Case 3"
  message: "Message 3"
"""
        )

        mock_responses = {
            "v1": {
                "extracted_features": {},
                "missing_features": [],
                "reply": "OK",
            }
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        results = evaluator.evaluate(versions=["v1"], mock_responses=mock_responses)

        assert len(results) == 3
        assert results[0]["case_name"] == "Case 1"
        assert results[1]["case_name"] == "Case 2"
        assert results[2]["case_name"] == "Case 3"

    def test_evaluate_compare_versions(self, tmp_path):
        """Should evaluate and allow comparison of versions."""
        eval_file = tmp_path / "cases.yaml"
        eval_file.write_text(
            """- name: "Quality test"
  message: "Good kitchen, excellent exterior"
"""
        )

        mock_responses = {
            "v1": {
                "extracted_features": {"KitchenQual": "Good"},
                "missing_features": ["OverallQual"],
                "reply": "V1 response",
            },
            "v2": {
                "extracted_features": {
                    "KitchenQual": "Good",
                    "ExterQual": "Excellent",
                },
                "missing_features": ["OverallQual"],
                "reply": "V2 response",
            },
        }

        evaluator = ExtractionEvaluator(str(eval_file))
        results = evaluator.evaluate(
            versions=["v1", "v2"], mock_responses=mock_responses
        )

        result = results[0]
        v1_extracted = result["results"]["v1"]["output"]["extracted_features"]
        v2_extracted = result["results"]["v2"]["output"]["extracted_features"]

        # v2 extracted more non-None features
        v1_count = sum(1 for v in v1_extracted.values() if v is not None)
        v2_count = sum(1 for v in v2_extracted.values() if v is not None)
        assert v2_count >= v1_count
