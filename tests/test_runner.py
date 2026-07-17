"""Unit tests for the training runner.

Validates that only a single training job runs concurrently, raising
TrainingError on collision and releasing lock resources upon completion.
"""

import time
from typing import Any
import pytest

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import TrainingError
from backend.schemas.datasets import DatasetMetadata
from backend.schemas.training import HyperparametersSchema, LoraConfigSchema, TrainingJobConfig, TrainingJobState
from backend.training.runner import TrainingRunner


class MockDatasetService:
    """Mock dataset registry client."""

    def __init__(self) -> None:
        self.db = {}

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        return self.db.get(dataset_id)


class MockStorageManager:
    """Mock storage layers for IO tests."""

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
    """Mock checkpoints catalog registry."""

    def __init__(self) -> None:
        self.jobs = {}
        self.checkpoints = {}

    def save_job_status(self, status: Any) -> None:
        self.jobs[status.job_id] = status

    def get_job_status(self, job_id: str) -> Any | None:
        return self.jobs.get(job_id)

    def save_checkpoint(self, entry: Any) -> None:
        self.checkpoints[entry.checkpoint_id] = entry


def test_runner_concurrency_lock() -> None:
    """Asserts that starting a second job fails with lock exceptions while a job is running."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()
    registry = MockModelRegistry()

    # Register mock READY dataset
    meta = DatasetMetadata(
        dataset_id="dataset_ready_hash",
        dataset_name="alpaca-dataset",
        raw_path="data/raw/data.json",
        lifecycle_state=DatasetLifecycleState.READY
    )
    ds_service.db["dataset_ready_hash"] = meta

    runner = TrainingRunner(storage, ds_service, registry)

    config1 = TrainingJobConfig(
        job_id="job_id_concurrency_1",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="dataset_ready_hash",
        hyperparameters=HyperparametersSchema(epochs=1),
        lora_config=LoraConfigSchema()
    )

    config2 = TrainingJobConfig(
        job_id="job_id_concurrency_2",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="dataset_ready_hash",
        hyperparameters=HyperparametersSchema(epochs=1),
        lora_config=LoraConfigSchema()
    )

    # Launch first job
    runner.start_training_job(config1)

    # Launching second job should fail with TrainingError due to acquired lock
    with pytest.raises(TrainingError) as exc_info:
        runner.start_training_job(config2)
    assert "occupied" in str(exc_info.value)

    # Wait for the first job to finish (epochs=1 -> 5 steps * 0.2s = 1s total duration)
    # Poll for completion with a timeout of 2.0 seconds
    completed = False
    for _ in range(20):
        time.sleep(0.1)
        status = runner.get_job_status("job_id_concurrency_1")
        if status and status.state == TrainingJobState.COMPLETED:
            completed = True
            break

    assert completed is True
    assert runner.get_job_status("job_id_concurrency_1").state == TrainingJobState.COMPLETED
