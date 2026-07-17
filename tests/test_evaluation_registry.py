"""Unit tests for the SQLite evaluation database registry extension.

Validates CRUD persistence for evaluation result objects and leaderboard benchmark entries.
"""

from backend.registry.catalog import SQLiteModelRegistry
from backend.schemas.evaluation import EvaluationResult, MetricSummary, TaskType


def test_sqlite_evaluation_registry_crud(tmp_path) -> None:
    """Asserts that evaluation saves, lists, and leaderboard filters function correctly in SQLite."""
    db_file = tmp_path / "models.db"
    registry = SQLiteModelRegistry(db_file)

    # 1. Assert registry is empty initially
    assert len(registry.list_evaluations()) == 0

    # 2. Save evaluation result
    result = EvaluationResult(
        evaluation_id="eval_test_uuid",
        model_id="checkpoint_uuid_456",
        dataset_id="dataset_uuid_789",
        task_type=TaskType.GENERATION,
        metrics=MetricSummary(bleu=0.88),
        runtime_seconds=5.25,
        report_path="artifacts/reports/eval_test_uuid/report.json"
    )
    registry.save_evaluation(result)

    # 3. Retrieve evaluation result
    retrieved = registry.get_evaluation("eval_test_uuid")
    assert retrieved is not None
    assert retrieved.evaluation_id == "eval_test_uuid"
    assert retrieved.metrics.bleu == 0.88
    assert len(registry.list_evaluations()) == 1

    # 4. Save and query benchmark leaderboard ranking entries
    registry.save_benchmark_entry(
        benchmark_id="bench_test_uuid",
        dataset_id="dataset_uuid_789",
        model_id="checkpoint_uuid_456",
        metric_name="bleu",
        metric_value=0.88
    )

    leaderboard = registry.get_leaderboard_entries("dataset_uuid_789")
    assert len(leaderboard) == 1
    assert leaderboard[0]["metric_value"] == 0.88
    assert leaderboard[0]["model_id"] == "checkpoint_uuid_456"
