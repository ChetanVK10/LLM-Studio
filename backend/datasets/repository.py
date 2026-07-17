"""Dataset Registry Persistence Repository.

Implements a pluggable database design utilizing a base interface IDatasetRepository
and a local SQLite SQLiteDatasetRepository implementation.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from backend.schemas.datasets import DatasetMetadata

logger = logging.getLogger("app")


class IDatasetRepository(ABC):
    """Abstract interface defining registry catalog CRUD operations."""

    @abstractmethod
    def save(self, metadata: DatasetMetadata) -> None:
        """Persists or updates a dataset metadata record.

        Args:
            metadata: DatasetMetadata Pydantic object.
        """
        pass

    @abstractmethod
    def get_by_id(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieves metadata matching the target ID key.

        Args:
            dataset_id: Cryptographic content hash (SHA-256).

        Returns:
            DatasetMetadata if found, None otherwise.
        """
        pass

    @abstractmethod
    def list_all(self) -> List[DatasetMetadata]:
        """Fetches all registered dataset records from the database.

        Returns:
            List of DatasetMetadata records.
        """
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Deletes a dataset entry matching the ID.

        Args:
            dataset_id: Target database record key.

        Returns:
            True if the record existed and was removed, False otherwise.
        """
        pass


class SQLiteDatasetRepository(IDatasetRepository):
    """SQLite implementation of dataset catalog registers."""

    def __init__(self, db_path: Path) -> None:
        """Initializes database files directory and table structures.

        Args:
            db_path: Path to database file location.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper connection manager.

        Returns:
            sqlite3.Connection.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Constructs tables layout on load if not existing."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets_registry (
                    dataset_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def save(self, metadata: DatasetMetadata) -> None:
        """Inserts or overwrites active dataset entry.

        Args:
            metadata: Target metadata.
        """
        serialized = metadata.model_dump_json()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets_registry
                (dataset_id, dataset_name, lifecycle_state, source, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.dataset_id,
                    metadata.dataset_name,
                    metadata.lifecycle_state.value,
                    metadata.source,
                    metadata.created_at.isoformat(),
                    serialized
                )
            )
            conn.commit()

    def get_by_id(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Resolves target database entries matching unique hash keys.

        Args:
            dataset_id: SHA-256 content signature string.

        Returns:
            DatasetMetadata if found, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM datasets_registry WHERE dataset_id = ?",
                (dataset_id,)
            ).fetchone()
            if row:
                try:
                    return DatasetMetadata.model_validate_json(row["metadata_json"])
                except Exception as e:
                    logger.error(
                        f"Database Registry: Failed to deserialize row for target '{dataset_id}': {e}"
                    )
                    return None
            return None

    def list_all(self) -> List[DatasetMetadata]:
        """Fetches all rows sorted by creation date descending.

        Returns:
            List of validated metadata elements.
        """
        datasets: List[DatasetMetadata] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT metadata_json FROM datasets_registry ORDER BY created_at DESC"
            ).fetchall()
            for row in rows:
                try:
                    datasets.append(DatasetMetadata.model_validate_json(row["metadata_json"]))
                except Exception as e:
                    logger.error(f"Database Registry: Skipping malformed metadata row entry: {e}")
        return datasets

    def delete(self, dataset_id: str) -> bool:
        """Deletes matching row entries.

        Args:
            dataset_id: Target key.

        Returns:
            True if records were deleted, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM datasets_registry WHERE dataset_id = ?",
                (dataset_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
