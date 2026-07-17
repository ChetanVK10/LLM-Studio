"""Experiment Tracker interface.

Placeholder module outlining future responsibilities for parameters and metrics logging
during training runs.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger("app")


class ExperimentTracker:
    """Manages active logging parameters, metrics charts steps, and remote syncing hooks."""

    def __init__(self) -> None:
        """Initializes experiment tracker interface."""
        logger.info("Initialized Experiment Tracker logger.")

    def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Logs single input configuration parameter or hyperparameter to the active run.

        Args:
            run_id: Active tracking run ID.
            key: Name identifier of parameter.
            value: Parameter value.
        """
        logger.info(f"Tracker: Logged param '{key}'='{value}' for run '{run_id}'")

    def log_metrics(self, run_id: str, step: int, metrics: Dict[str, Any]) -> None:
        """Logs a batch dictionary of step metrics values.

        Args:
            run_id: Active tracking run ID.
            step: Training epoch or optimization step index.
            metrics: Metric name-value mapping.
        """
        logger.info(f"Tracker: Logged metrics at step {step} for run '{run_id}': {metrics}")
