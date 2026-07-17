"""Unit tests for the evaluation pipeline.

Validates model readiness checks, dataset state validation, and report file generation loops.
"""

import json
import pytest
from typing import Any

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import EvaluationError
from backend.evaluation.pipeline import EvaluationPipeline
from backend.schemas.datasets import DatasetMetadata
from backend.schemas.evaluation import EvaluationConfig, TaskType
from backend.schemas.registry import RegistryEntry


class MockDatasetService:
    """Mock dataset registry database."""

    def __init__(self) -> None:
        self.db = {}

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        return self.db.get(dataset_id)


class MockStorageManager:
    """Mock storage layers."""

    def __init__(self) -> None:
        self.workspace_root = "/mock"
        self.db = {}

    def write_file(self, path: str, content: bytes | str) -> str:
        self.db[path] = content
        return path

    def read_file(self, path: str) -> bytes:
        val = self.db.get(path, b"")
        if isinstance(val, str):
            return val.encode("utf-8")
        return val

    def exists(self, path: str) -> bool:
        return path in self.db

    def ensure_dir(self, path: str) -> None:
        pass


class MockModelRegistry:
    """Mock database registry model catalog."""

    def __init__(self) -> None:
        self.checkpoints = {}
        self.evaluations = {}
        self.benchmarks = {}

    def get_checkpoint(self, checkpoint_id: str) -> RegistryEntry | None:
        return self.checkpoints.get(checkpoint_id)

    def save_evaluation(self, result: Any) -> None:
        self.evaluations[result.evaluation_id] = result

    def save_benchmark_entry(
        self,
        benchmark_id: str,
        dataset_id: str,
        model_id: str,
        metric_name: str,
        metric_value: float
    ) -> None:
        self.benchmarks[benchmark_id] = (dataset_id, model_id, metric_name, metric_value)


def test_evaluation_pipeline_validation_missing_model() -> None:
    """Asserts that trying to evaluate unregistered model checkpoints raises EvaluationError."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()
    registry = MockModelRegistry()

    pipeline = EvaluationPipeline(storage, ds_service, registry)
    config = EvaluationConfig(
        evaluation_id="eval_uuid_001",
        model_id="absent_model_id",
        dataset_id="dataset_ready_id",
        task_type=TaskType.GENERATION
    )

    with pytest.raises(EvaluationError) as exc_info:
        pipeline.run_evaluation(config)
    assert "checkpoint" in str(exc_info.value).lower()


def test_evaluation_pipeline_validation_missing_dataset() -> None:
    """Asserts that running evaluations on missing datasets raises EvaluationError."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()
    registry = MockModelRegistry()

    registry.checkpoints["model_id_123"] = RegistryEntry(
        checkpoint_id="model_id_123",
        model_name="test-llama",
        base_model="llama",
        path="path/to/model"
    )

    pipeline = EvaluationPipeline(storage, ds_service, registry)
    config = EvaluationConfig(
        evaluation_id="eval_uuid_001",
        model_id="model_id_123",
        dataset_id="absent_dataset_id",
        task_type=TaskType.GENERATION
    )

    with pytest.raises(EvaluationError) as exc_info:
        pipeline.run_evaluation(config)
    assert "dataset" in str(exc_info.value).lower()


def test_evaluation_pipeline_success() -> None:
    """Asserts that evaluation successfully completes when both checkpoints and datasets are READY."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()
    registry = MockModelRegistry()

    # 1. Setup target model checkpoint and READY dataset
    registry.checkpoints["model_id_123"] = RegistryEntry(
        checkpoint_id="model_id_123",
        model_name="test-llama",
        base_model="llama",
        path="path/to/model"
    )

    ds_service.db["dataset_ready_id"] = DatasetMetadata(
        dataset_id="dataset_ready_id",
        dataset_name="evaluation-alpaca",
        raw_path="data/raw/data.json",
        lifecycle_state=DatasetLifecycleState.READY
    )

    # 2. Write validation splits to mock storage
    samples = '{"instruction": "Calculate 1+1", "output": "2"}\n'
    storage.write_file("data/processed/dataset_ready_id/validation.jsonl", samples.encode("utf-8"))

    pipeline = EvaluationPipeline(storage, ds_service, registry)
    config = EvaluationConfig(
        evaluation_id="eval_uuid_001",
        model_id="model_id_123",
        dataset_id="dataset_ready_id",
        task_type=TaskType.GENERATION
    )

    # 3. Run pipeline
    result = pipeline.run_evaluation(config)

    assert result.evaluation_id == "eval_uuid_001"
    assert result.metrics.bleu == 1.0
    assert len(registry.evaluations) == 1
    assert "bench_eval_uuid_001" in registry.benchmarks
