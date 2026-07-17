"""Unit tests for Pydantic schemas.

Validates that data contracts perform type enforcement and parameter checking as expected.
"""

import pytest
from pydantic import ValidationError

from backend.schemas.training import TrainingRequest
from backend.schemas.evaluation import EvaluationResult


def test_training_request_validation() -> None:
    """Verifies that TrainingRequest validates inputs successfully."""
    # Valid arguments
    req = TrainingRequest(
        model_name="meta-llama/Llama-3-8b",
        dataset_name="alpaca-cleaned",
        epochs=3,
        learning_rate=2e-4
    )
    assert req.epochs == 3
    assert req.learning_rate == 0.0002
    assert "q_proj" in req.target_modules

    # Missing required argument
    with pytest.raises(ValidationError):
        # model_name is required
        TrainingRequest(dataset_name="alpaca-cleaned")


def test_evaluation_result_validation() -> None:
    """Verifies that EvaluationResult parses floats and dictionaries fields correctly."""
    result = EvaluationResult(
        eval_id="eval_001",
        model_name="llama3-adapter",
        dataset_name="gsm8k",
        rouge_scores={"rougeL": 0.415},
        bertscore=0.88,
        exact_match=0.12,
        latency_ms=85.2,
        throughput_tokens_per_sec=42.1,
        llm_judge_score=4.5,
        cost_estimation_usd=0.012
    )
    
    assert result.eval_id == "eval_001"
    assert result.rouge_scores["rougeL"] == 0.415
    assert result.latency_ms == 85.2
