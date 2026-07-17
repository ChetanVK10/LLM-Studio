"""Unit tests for the benchmarking engine.

Verifies that the leaderboard aggregation lists sorted checkpoints correctly
and resolves models references in SQLite catalog indexes.
"""

from typing import Any, List

from backend.evaluation.benchmark import BenchmarkEngine
from backend.schemas.registry import RegistryEntry


class MockModelRegistry:
    """Mock checkpoints and benchmarks lookup registries."""

    def __init__(self) -> None:
        self.checkpoints = {}
        self.benchmarks = []

    def get_checkpoint(self, checkpoint_id: str) -> RegistryEntry | None:
        return self.checkpoints.get(checkpoint_id)

    def get_leaderboard_entries(self, dataset_id: str) -> List[dict]:
        return [b for b in self.benchmarks if b["dataset_id"] == dataset_id]


def test_benchmark_engine_leaderboard() -> None:
    """Asserts that leaderboard ranks are sorted by score descending and deduplicated by model ID."""
    registry = MockModelRegistry()

    # 1. Setup mock checkpoints
    registry.checkpoints["model_uuid_1"] = RegistryEntry(
        checkpoint_id="model_uuid_1",
        model_name="llama-fine-tuned-1",
        base_model="llama-8b",
        path="paths/model_1"
    )
    registry.checkpoints["model_uuid_2"] = RegistryEntry(
        checkpoint_id="model_uuid_2",
        model_name="llama-fine-tuned-2",
        base_model="llama-8b",
        path="paths/model_2"
    )

    # 2. Append mock runs records
    registry.benchmarks.append({
        "model_id": "model_uuid_1",
        "dataset_id": "dataset_test_id",
        "metric_name": "bleu",
        "metric_value": 0.82,
        "evaluated_at": "2026-07-17T12:00:00"
    })
    
    registry.benchmarks.append({
        "model_id": "model_uuid_2",
        "dataset_id": "dataset_test_id",
        "metric_name": "bleu",
        "metric_value": 0.94,
        "evaluated_at": "2026-07-17T12:30:00"
    })

    # Add a lower score run for model_uuid_1 to verify deduplication holds the best run
    registry.benchmarks.append({
        "model_id": "model_uuid_1",
        "dataset_id": "dataset_test_id",
        "metric_name": "bleu",
        "metric_value": 0.75,
        "evaluated_at": "2026-07-16T12:00:00"
    })

    engine = BenchmarkEngine(registry)
    leaderboard = engine.get_leaderboard("dataset_test_id")

    # 3. Assert sorting and deduplication
    assert len(leaderboard) == 2
    
    # Best model is model_uuid_2
    assert leaderboard[0]["model_id"] == "model_uuid_2"
    assert leaderboard[0]["metric_value"] == 0.94
    assert leaderboard[0]["model_name"] == "llama-fine-tuned-2"

    # Second is model_uuid_1 (holding the 0.82 best score, not the 0.75)
    assert leaderboard[1]["model_id"] == "model_uuid_1"
    assert leaderboard[1]["metric_value"] == 0.82
    assert leaderboard[1]["model_name"] == "llama-fine-tuned-1"
