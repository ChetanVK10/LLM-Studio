"""Unit tests for the model evaluator.

Verifies that model evaluator generates outputs matching validation targets.
"""

from backend.evaluation.evaluator import ModelEvaluator


def test_model_evaluator_generation() -> None:
    """Asserts model evaluator returns formatted prediction arrays."""
    evaluator = ModelEvaluator(model_id="test_model_id", checkpoint_dir="mock_path")
    samples = [
        {"instruction": "Calculate 2+2", "output": "4"},
        {"instruction": "Capital of France", "output": "Paris"}
    ]

    predictions = evaluator.generate_predictions(samples, max_new_tokens=64, temperature=0.0)

    assert len(predictions) == 2
    assert predictions[0] == "4"
    assert predictions[1] == "Paris"
