"""Benchmarking and Leaderboard Engine.

Aggregates multiple evaluation results to rank fine-tuned models
and checkpoints relative to target metrics and runtimes.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("app")


class BenchmarkEngine:
    """Aggregates evaluation database logs to build comparative leaderboards."""

    def __init__(self, model_registry: Any) -> None:
        """Initializes benchmark engine.

        Args:
            model_registry: SQLite model registry DB.
        """
        self.model_registry = model_registry

    def get_leaderboard(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Queries registry catalogs and aggregates scores to rank models.

        Args:
            dataset_id: Target dataset registry ID.

        Returns:
            List of sorted comparison dictionaries.
        """
        raw_entries = self.model_registry.get_leaderboard_entries(dataset_id)
        if not raw_entries:
            return []

        leaderboard = []
        for entry in raw_entries:
            model_id = entry["model_id"]
            checkpoint = self.model_registry.get_checkpoint(model_id)

            model_name = checkpoint.model_name if checkpoint else f"checkpoint-{model_id[:8]}"
            base_model = checkpoint.base_model if checkpoint else "N/A"

            leaderboard.append({
                "model_id": model_id,
                "model_name": model_name,
                "base_model": base_model,
                "metric_name": entry["metric_name"].upper(),
                "metric_value": round(entry["metric_value"], 4),
                "evaluated_at": entry["evaluated_at"]
            })

        # Deduplicate so that only the best metric run displays per model_id
        seen = set()
        deduplicated = []
        
        # Sort by score descending to keep the best run
        sorted_runs = sorted(leaderboard, key=lambda x: x["metric_value"], reverse=True)
        for run in sorted_runs:
            if run["model_id"] not in seen:
                seen.add(run["model_id"])
                deduplicated.append(run)

        return deduplicated
