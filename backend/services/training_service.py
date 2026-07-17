"""Training Orchestration Service.

Validates configurations, manages locks, dispatches background thread runners,
and fetches database registries histories.
"""

import logging
from typing import List, Optional

from backend.registry.catalog import SQLiteModelRegistry
from backend.schemas.registry import RegistryEntry
from backend.schemas.training import TrainingJobConfig, TrainingJobStatus, TrainingRequest, TrainingResult
from backend.training.runner import TrainingRunner

logger = logging.getLogger("app")


class TrainingService:
    """Orchestrates local training loops configurations, threads, and databases catalogs."""

    def __init__(self, runner: TrainingRunner, registry: SQLiteModelRegistry) -> None:
        """Initializes the training service.

        Args:
            runner: Background thread worker runner.
            registry: SQLite persistence catalog repository.
        """
        self.runner = runner
        self.registry = registry

    def start_training(self, config: TrainingJobConfig) -> str:
        """Audits readiness parameters and dispatches local training runner.

        Args:
            config: Target job configurations settings.

        Returns:
            Launched job identifier key.
        """
        logger.info(f"TrainingService: Dispatched training launch for job '{config.job_id}'")
        return self.runner.start_training_job(config)

    def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        """Resolves runtime step progress from active memory files or database records.

        Args:
            job_id: Unique job run ID.

        Returns:
            TrainingJobStatus metadata if logged, None otherwise.
        """
        return self.runner.get_job_status(job_id)

    def list_training_history(self) -> List[TrainingJobStatus]:
        """Fetches list of all previous fine-tuning jobs statuses from database.

        Returns:
            List of TrainingJobStatus items.
        """
        return self.registry.list_job_statuses()

    def list_checkpoints(self) -> List[RegistryEntry]:
        """Fetches cataloged adapter checkpoints list from registry databases.

        Returns:
            List of RegistryEntry checkpoints metadata.
        """
        return self.registry.list_checkpoints()

    # --- Legacy Compatibility Methods ---

    def trigger_training(self, request: TrainingRequest) -> TrainingResult:
        """Legacy compatibility wrapper triggering mock results logs.

        Args:
            request: Compatibility wrapper request.

        Returns:
            TrainingResult mock details.
        """
        logger.info(f"TrainingService: Launching mock run for base model '{request.model_name}'")
        return TrainingResult(
            run_id="run_qlora_llama_001",
            status="completed",
            output_checkpoint_dir="artifacts/checkpoints/qlora_llama_001",
            metrics={"final_loss": 0.45, "epochs_completed": request.epochs}
        )

    def list_active_jobs(self) -> List[TrainingResult]:
        """Legacy compatibility list method.

        Returns:
            Empty logs list.
        """
        return []
