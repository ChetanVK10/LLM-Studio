"""Unit tests for the pre-training validator.

Validates that model format checks, dataset lifecycle READY checks,
path confirmations, and hyperparameter range checks are correctly audited.
"""

import pytest

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import DatasetValidationError, HyperparameterError, TrainingError
from backend.schemas.datasets import DatasetMetadata
from backend.schemas.training import HyperparametersSchema, LoraConfigSchema, TrainingJobConfig
from backend.training.validator import PreTrainingValidator


class MockDatasetService:
    """Mock dataset service registry helper."""

    def __init__(self) -> None:
        self.db = {}

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        return self.db.get(dataset_id)


class MockStorageManager:
    """Mock storage manager helper."""

    def __init__(self) -> None:
        self.workspace_root = "/mock"

    def ensure_dir(self, path: str) -> None:
        pass


def test_pre_training_validator_success() -> None:
    """Asserts that standard correct configuration configs validate successfully."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()

    # Register valid READY dataset
    meta = DatasetMetadata(
        dataset_id="valid_sha256_hash",
        dataset_name="clean_dataset",
        raw_path="data/raw/clean_dataset.json",
        lifecycle_state=DatasetLifecycleState.READY
    )
    ds_service.db["valid_sha256_hash"] = meta

    config = TrainingJobConfig(
        job_id="job_uuid_123",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="valid_sha256_hash",
        hyperparameters=HyperparametersSchema(epochs=3, batch_size=4, learning_rate=2e-4),
        lora_config=LoraConfigSchema(r=8, lora_alpha=16, target_modules=["q_proj"])
    )

    # Validate readiness (should pass without raise)
    PreTrainingValidator.validate_readiness(config, ds_service, storage)


def test_pre_training_validator_missing_dataset() -> None:
    """Asserts that validating unregistered datasets raises DatasetValidationError."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()

    config = TrainingJobConfig(
        job_id="job_uuid_123",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="absent_dataset_hash",
        hyperparameters=HyperparametersSchema(),
        lora_config=LoraConfigSchema()
    )

    with pytest.raises(DatasetValidationError) as exc_info:
        PreTrainingValidator.validate_readiness(config, ds_service, storage)
    
    assert "not registered in catalog" in str(exc_info.value)


def test_pre_training_validator_not_ready_dataset() -> None:
    """Asserts that validating datasets that are not READY raises DatasetValidationError."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()

    # Register dataset in UPLOADED state
    meta = DatasetMetadata(
        dataset_id="valid_sha256_hash",
        dataset_name="clean_dataset",
        raw_path="data/raw/clean_dataset.json",
        lifecycle_state=DatasetLifecycleState.UPLOADED
    )
    ds_service.db["valid_sha256_hash"] = meta

    config = TrainingJobConfig(
        job_id="job_uuid_123",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="valid_sha256_hash",
        hyperparameters=HyperparametersSchema(),
        lora_config=LoraConfigSchema()
    )

    with pytest.raises(DatasetValidationError) as exc_info:
        PreTrainingValidator.validate_readiness(config, ds_service, storage)
        
    assert "not ready for training" in str(exc_info.value)


def test_pre_training_validator_invalid_hyperparams() -> None:
    """Asserts that out-of-range optimization parameters raise HyperparameterError."""
    ds_service = MockDatasetService()
    storage = MockStorageManager()

    meta = DatasetMetadata(
        dataset_id="valid_sha256_hash",
        dataset_name="clean_dataset",
        raw_path="data/raw/clean_dataset.json",
        lifecycle_state=DatasetLifecycleState.READY
    )
    ds_service.db["valid_sha256_hash"] = meta

    # Set empty target modules to bypass Pydantic constructor checks but hit validator audit
    config = TrainingJobConfig(
        job_id="job_uuid_123",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        dataset_id="valid_sha256_hash",
        hyperparameters=HyperparametersSchema(),
        lora_config=LoraConfigSchema(target_modules=[])
    )

    with pytest.raises(HyperparameterError) as exc_info:
        PreTrainingValidator.validate_readiness(config, ds_service, storage)
        
    assert "target modules list cannot be empty" in str(exc_info.value)
