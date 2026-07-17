"""Unit tests for dataset caching.

Validates cache hits and misses by checking repository entries for identical
SHA-256 signatures.
"""

from backend.core.constants import DatasetLifecycleState
from backend.datasets.cache import DatasetCacheManager
from backend.schemas.datasets import DatasetMetadata


class MockDatasetRepository:
    """Mock repository mimicking CRUD interfaces for testing."""

    def __init__(self) -> None:
        self.db = {}

    def save(self, metadata: DatasetMetadata) -> None:
        self.db[metadata.dataset_id] = metadata

    def get_by_id(self, dataset_id: str) -> DatasetMetadata | None:
        return self.db.get(dataset_id)

    def list_all(self) -> list[DatasetMetadata]:
        return list(self.db.values())

    def delete(self, dataset_id: str) -> bool:
        return self.db.pop(dataset_id, None) is not None


def test_cache_hit_and_miss() -> None:
    """Asserts cache lookup results match registered metadata hashes."""
    repo = MockDatasetRepository()
    cache_manager = DatasetCacheManager(repo)
    
    # 1. Verification of cache miss
    assert cache_manager.check_cache_hit("non_existent_hash") is None

    # 2. Register mock entry
    meta = DatasetMetadata(
        dataset_id="sha256_mocked_hash_key",
        dataset_name="alpaca-data",
        raw_path="data/raw/data.json",
        lifecycle_state=DatasetLifecycleState.READY
    )
    repo.save(meta)

    # 3. Verification of cache hit
    hit = cache_manager.check_cache_hit("sha256_mocked_hash_key")
    assert hit is not None
    assert hit.dataset_name == "alpaca-data"
    assert hit.lifecycle_state == DatasetLifecycleState.READY
