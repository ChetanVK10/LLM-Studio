"""Dataset Preprocessing Cache Module.

Looks up dataset records based on content hashes to identify redundant uploads,
minimizing repeated file processing and validation workloads.
"""

from typing import Optional

from backend.datasets.repository import IDatasetRepository
from backend.schemas.datasets import DatasetMetadata


class DatasetCacheManager:
    """Verifies content duplicates and routes tasks to cached processed files."""

    def __init__(self, repository: IDatasetRepository) -> None:
        """Initializes cache checker.

        Args:
            repository: Catalog persistence layer instance.
        """
        self.repository = repository

    def check_cache_hit(self, content_hash: str) -> Optional[DatasetMetadata]:
        """Queries registry repository database for matching SHA-256 signatures.

        Args:
            content_hash: SHA-256 hash checksum representing dataset content.

        Returns:
            Matched DatasetMetadata if registered, None otherwise.
        """
        return self.repository.get_by_id(content_hash)
