"""Run Lifecycle Manager.

Placeholder module outlining future responsibilities for run lifecycle states management,
UUID creations, and duration timings.
"""

from datetime import datetime
import logging
from typing import Any, Dict

from backend.schemas.experiments import ExperimentMetadata

logger = logging.getLogger("app")


class RunLifecycleManager:
    """Orchestrates starting, stopping, failing, and cancelling tracking runs sessions."""

    def __init__(self) -> None:
        """Initializes lifecycle manager."""
        logger.info("Initialized Experiment Runs Lifecycle Manager.")

    def start_run(self, experiment_id: str, name: str, parameters: Dict[str, Any]) -> ExperimentMetadata:
        """Instantiates and registers a new active experiment run.

        Args:
            experiment_id: Target grouping ID.
            name: Friendly display name.
            parameters: Start configuration variables.

        Returns:
            ExperimentMetadata representing tracking run data.
        """
        run_id = "run_session_mock_hash_001"
        logger.info(
            f"Runs: Starting run '{run_id}' under experiment category: '{experiment_id}'"
        )
        return ExperimentMetadata(
            experiment_id=experiment_id,
            run_id=run_id,
            name=name,
            status="running",
            parameters=parameters,
            metrics={},
            created_at=datetime.utcnow()
        )

    def complete_run(self, run_id: str, final_metrics: Dict[str, Any]) -> None:
        """Sets run status to completed and locks parameters mapping.

        Args:
            run_id: Target run session.
            final_metrics: Dict of calculated metrics results.
        """
        logger.info(f"Runs: Completing run '{run_id}' with final metrics: {final_metrics}")
