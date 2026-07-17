"""Evaluation and Leaderboard Service.

Exposes interfaces to execute evaluation pipeline runs, query historical reports,
and retrieve aggregated leaderboard ranks.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from backend.evaluation.benchmark import BenchmarkEngine
from backend.evaluation.pipeline import EvaluationPipeline
from backend.evaluation.report import ReportGenerator
from backend.schemas.evaluation import EvaluationConfig, EvaluationResult

logger = logging.getLogger("app")


class EvaluationService:
    """Orchestrates model evaluation pipelines, leaderboard updates, and Markdown reports generation."""

    def __init__(
        self,
        pipeline: EvaluationPipeline,
        benchmark: BenchmarkEngine,
        model_registry: Any,
        storage_manager: Any
    ) -> None:
        """Initializes service.

        Args:
            pipeline: Pipeline run executor.
            benchmark: Aggregator of comparative leaderboards.
            model_registry: Checkpoint catalog database catalog.
            storage_manager: Local filesystem storage helper.
        """
        self.pipeline = pipeline
        self.benchmark = benchmark
        self.model_registry = model_registry
        self.storage_manager = storage_manager

    def run_evaluation(self, config: EvaluationConfig) -> EvaluationResult:
        """Launches the validation checks, inference loop, and registers results.

        Args:
            config: Job configuration settings.

        Returns:
            EvaluationResult summary dataset.
        """
        logger.info(f"EvaluationService: Starting evaluation run '{config.evaluation_id}'")
        return self.pipeline.run_evaluation(config)

    def get_evaluation(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Queries database for evaluation results records.

        Args:
            evaluation_id: Unique evaluation identifier.

        Returns:
            EvaluationResult if located, None otherwise.
        """
        return self.model_registry.get_evaluation(evaluation_id)

    def list_evaluations(self) -> List[EvaluationResult]:
        """Queries list of all historical evaluation reports.

        Returns:
            List of EvaluationResult.
        """
        return self.model_registry.list_evaluations()

    def get_leaderboard(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Compiles metric performance leaderboards comparing models on a dataset.

        Args:
            dataset_id: Target dataset registry key ID.

        Returns:
            Aggregated rankings list.
        """
        return self.benchmark.get_leaderboard(dataset_id)

    def get_report_markdown(self, evaluation_id: str) -> str:
        """Compiles details from report.json and generates Markdown.

        Args:
            evaluation_id: Target evaluation runs.

        Returns:
            String containing formatted Markdown text.
        """
        report_json_path = f"artifacts/reports/{evaluation_id}/report.json"
        if not self.storage_manager.exists(report_json_path):
            return f"Error: Unable to locate report output files for evaluation ID `{evaluation_id}`."

        try:
            raw_data = self.storage_manager.read_file(report_json_path)
            data = json.loads(raw_data.decode("utf-8"))
            
            result = self.get_evaluation(evaluation_id)
            if not result:
                return "Error: Evaluation record not found in database registry catalog."

            return ReportGenerator.generate_markdown_report(result, data.get("samples", []))
        except Exception as e:
            logger.error(f"EvaluationService: Failed to compile Markdown report: {e}")
            return f"Error: Failed to parse report files: {e}"
