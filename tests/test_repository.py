"""Unit tests for the SQLite dataset repository.

Validates that SQLite database CRUD operations functions correctly on temp databases,
serializing and deserializing Pydantic schemas successfully.
"""

from backend.core.constants import DatasetLifecycleState
from backend.datasets.repository import SQLiteDatasetRepository
from backend.schemas.datasets import DatasetMetadata


def test_sqlite_repository_crud(tmp_path) -> None:
    """Verifies that save, retrieve, list, update, and delete SQL operations execute correctly."""
    db_file = tmp_path / "test_datasets.db"
    repo = SQLiteDatasetRepository(db_file)

    # 1. Verification of empty list
    assert len(repo.list_all()) == 0

    # 2. Save metadata record
    meta = DatasetMetadata(
        dataset_id="unique_sha256_hash",
        dataset_name="alpaca-small",
        raw_path="data/raw/unique_sha256_hash.json",
        lifecycle_state=DatasetLifecycleState.UPLOADED
    )
    repo.save(meta)

    # 3. Retrieve metadata record
    retrieved = repo.get_by_id("unique_sha256_hash")
    assert retrieved is not None
    assert retrieved.dataset_name == "alpaca-small"
    assert retrieved.lifecycle_state == DatasetLifecycleState.UPLOADED

    # 4. List all records
    records = repo.list_all()
    assert len(records) == 1
    assert records[0].dataset_id == "unique_sha256_hash"

    # 5. Update metadata record status
    meta.lifecycle_state = DatasetLifecycleState.READY
    repo.save(meta)
    updated = repo.get_by_id("unique_sha256_hash")
    assert updated.lifecycle_state == DatasetLifecycleState.READY

    # 6. Delete metadata record
    assert repo.delete("unique_sha256_hash") is True
    assert repo.get_by_id("unique_sha256_hash") is None
    assert len(repo.list_all()) == 0
