"""Unit tests for the SQLite model checkpoint registry.

Validates catalog registration operations, saving and retrieving job status logs,
and serializing configuration dictionaries in SQLite databases.
"""

from backend.registry.catalog import SQLiteModelRegistry
from backend.schemas.registry import RegistryEntry
from backend.schemas.training import TrainingJobState, TrainingJobStatus


def test_sqlite_model_registry_crud(tmp_path) -> None:
    """Verifies that save, retrieve, list, and state update catalog actions run correctly."""
    db_file = tmp_path / "models.db"
    registry = SQLiteModelRegistry(db_file)

    # 1. Assert lists are empty initially
    assert len(registry.list_checkpoints()) == 0
    assert len(registry.list_job_statuses()) == 0

    # 2. Save training job status
    status = TrainingJobStatus(
        job_id="job_status_123",
        state=TrainingJobState.RUNNING
    )
    registry.save_job_status(status)

    # 3. Retrieve training job status
    retrieved_status = registry.get_job_status("job_status_123")
    assert retrieved_status is not None
    assert retrieved_status.state == TrainingJobState.RUNNING

    # 4. Save model checkpoint adapter metadata
    entry = RegistryEntry(
        checkpoint_id="adapter_chk_001",
        model_name="llama3-lora-checkpoint",
        base_model="meta-llama/Meta-Llama-3-8B",
        path="artifacts/checkpoints/adapter_chk_001",
        tags=["qlora", "alpaca"],
        hyperparameters={"epochs": 3, "learning_rate": 2e-4}
    )
    registry.save_checkpoint(entry)

    # 5. Retrieve model checkpoint adapter
    retrieved_checkpoint = registry.get_checkpoint("adapter_chk_001")
    assert retrieved_checkpoint is not None
    assert retrieved_checkpoint.model_name == "llama3-lora-checkpoint"
    assert retrieved_checkpoint.tags == ["qlora", "alpaca"]
    assert retrieved_checkpoint.hyperparameters.get("epochs") == 3

    # 6. List all records
    assert len(registry.list_checkpoints()) == 1
    assert len(registry.list_job_statuses()) == 1
