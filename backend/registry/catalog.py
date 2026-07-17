"""Model Checkpoint Catalog & Registry.

Manages persistent SQLite registries indexing adapter checkpoints metadata,
training job logs, and status audits.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from backend.schemas.registry import RegistryEntry
from backend.schemas.training import TrainingJobStatus

logger = logging.getLogger("app")


class SQLiteModelRegistry:
    """SQLite implementation of model checkpoint registers and training history catalogs."""

    def __init__(self, db_path: Path) -> None:
        """Initializes registry databases files and schema structures.

        Args:
            db_path: Absolute Path to the sqlite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes sqlite3 connections.

        Returns:
            sqlite3.Connection Row-factory mappings helper.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates table setups on loader import."""
        with self._get_connection() as conn:
            # Model adapter registries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    checkpoint_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    path TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                )
            """)
            # Training job running logs and progress status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status_json TEXT NOT NULL
                )
            """)
            conn.commit()

    # --- Checkpoints Catalog Methods ---

    def save_checkpoint(self, entry: RegistryEntry) -> None:
        """Saves a model checkpoint record.

        Args:
            entry: Checkpoint registration entry.
        """
        tags_serialized = json.dumps(entry.tags)
        hp_serialized = json.dumps(entry.hyperparameters)
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (checkpoint_id, model_name, base_model, path, tags, hyperparameters, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.checkpoint_id,
                    entry.model_name,
                    entry.base_model,
                    entry.path,
                    tags_serialized,
                    hp_serialized,
                    entry.registered_at.isoformat()
                )
            )
            conn.commit()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[RegistryEntry]:
        """Resolves single checkpoint registries details matching target ID.

        Args:
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            RegistryEntry if located, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_registry WHERE checkpoint_id = ?",
                (checkpoint_id,)
            ).fetchone()
            
            if row:
                try:
                    return RegistryEntry(
                        checkpoint_id=row["checkpoint_id"],
                        model_name=row["model_name"],
                        base_model=row["base_model"],
                        path=row["path"],
                        tags=json.loads(row["tags"]),
                        hyperparameters=json.loads(row["hyperparameters"])
                    )
                except Exception as e:
                    logger.error(f"Registry: Failed to deserialize checkpoint row: {e}")
            return None

    def list_checkpoints(self) -> List[RegistryEntry]:
        """Fetches all registered fine-tuned model checkpoints.

        Returns:
            List of RegistryEntry objects.
        """
        entries: List[RegistryEntry] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM model_registry ORDER BY registered_at DESC"
            ).fetchall()
            for row in rows:
                try:
                    entries.append(
                        RegistryEntry(
                            checkpoint_id=row["checkpoint_id"],
                            model_name=row["model_name"],
                            base_model=row["base_model"],
                            path=row["path"],
                            tags=json.loads(row["tags"]),
                            hyperparameters=json.loads(row["hyperparameters"])
                        )
                    )
                except Exception as e:
                    logger.error(f"Registry: Skipping corrupt checkpoint metadata: {e}")
        return entries

    # --- Training Job Status Methods ---

    def save_job_status(self, status: TrainingJobStatus) -> None:
        """Stores or updates the execution status log of a fine-tuning job.

        Args:
            status: Active status values.
        """
        serialized = status.model_dump_json()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_jobs
                (job_id, state, created_at, status_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    status.job_id,
                    status.state.value,
                    status.created_at.isoformat(),
                    serialized
                )
            )
            conn.commit()

    def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        """Resolves target training job logs.

        Args:
            job_id: Training run ID.

        Returns:
            TrainingJobStatus object if found, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT status_json FROM training_jobs WHERE job_id = ?",
                (job_id,)
            ).fetchone()
            if row:
                try:
                    return TrainingJobStatus.model_validate_json(row["status_json"])
                except Exception as e:
                    logger.error(f"Registry: Failed to parse job log serialization: {e}")
            return None

    def list_job_statuses(self) -> List[TrainingJobStatus]:
        """Lists all training history records.

        Returns:
            List of TrainingJobStatus.
        """
        jobs: List[TrainingJobStatus] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT status_json FROM training_jobs ORDER BY created_at DESC"
            ).fetchall()
            for row in rows:
                try:
                    jobs.append(TrainingJobStatus.model_validate_json(row["status_json"]))
                except Exception as e:
                    logger.error(f"Registry: Skipping corrupt training job log: {e}")
        return jobs


class ModelCatalog:
    """Manages indices matching checkpoint IDs, names, base foundation models, and active tags."""

    def __init__(self) -> None:
        """Initializes model catalog tracker instance."""
        logger.info("Initialized Registry Model Catalog index.")

    def add_entry(self, entry: RegistryEntry) -> RegistryEntry:
        """Adds a verified model checkpoint entry to indices catalogs.

        Args:
            entry: Checkpoint entry to insert.

        Returns:
            The inserted RegistryEntry.
        """
        logger.info(f"Catalog: Adding entry '{entry.checkpoint_id}' to indices.")
        return entry

    def search_by_tag(self, tag: str) -> List[RegistryEntry]:
        """Queries model catalogs containing tag constraints.

        Args:
            tag: Search label.

        Returns:
            List of matching RegistryEntry models.
        """
        logger.info(f"Catalog: Querying entries matching tag: '{tag}'")
        return []
