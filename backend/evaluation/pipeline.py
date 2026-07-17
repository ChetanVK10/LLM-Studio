"""Evaluation Pipeline Orchestrator.

Loads model checkpoints, dataset splits, executes inferences, computes target metrics,
logs database results, and triggers structured report output formats.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import EvaluationError
from backend.evaluation.evaluator import ModelEvaluator
from backend.evaluation.metrics import MetricsEvaluator
from backend.schemas.evaluation import EvaluationConfig, EvaluationResult, MetricSummary, TaskType


class EvaluationPipeline:
    """Orchestrates validation, loading, inference, metrics calculation, and database registration."""

    def __init__(
        self,
        storage_manager: Any,
        dataset_service: Any,
        model_registry: Any
    ) -> None:
        """Initializes pipeline.

        Args:
            storage_manager: Read/write disk operations manager.
            dataset_service: Dataset catalog controller.
            model_registry: SQLite model registry.
        """
        self.storage_manager = storage_manager
        self.dataset_service = dataset_service
        self.model_registry = model_registry

    def run_evaluation(self, config: EvaluationConfig) -> EvaluationResult:
        """Executes the full evaluation pipeline.

        Args:
            config: Target EvaluationConfig configurations.

        Returns:
            EvaluationResult summarizations.

        Raises:
            EvaluationError: If validations fail or processing hits exceptions.
        """
        start_time = time.time()

        # 1. Validation Checks
        # Validate model checkpoint existence
        checkpoint = self.model_registry.get_checkpoint(config.model_id)
        if not checkpoint:
            raise EvaluationError(
                f"Evaluation failed: Model checkpoint '{config.model_id}' is not registered."
            )

        # Validate dataset existence and lifecycle state
        dataset = self.dataset_service.get_dataset(config.dataset_id)
        if not dataset:
            raise EvaluationError(
                f"Evaluation failed: Dataset '{config.dataset_id}' is not registered."
            )
        if dataset.lifecycle_state != DatasetLifecycleState.READY:
            raise EvaluationError(
                f"Evaluation failed: Dataset '{dataset.dataset_name}' is not in READY state."
            )

        # 2. Read evaluation split records (Validation or Train split)
        # Check processed path folders using storage manager
        validation_file = f"data/processed/{config.dataset_id}/validation.jsonl"
        train_file = f"data/processed/{config.dataset_id}/train.jsonl"
        
        target_file = None
        if self.storage_manager.exists(validation_file):
            target_file = validation_file
        elif self.storage_manager.exists(train_file):
            target_file = train_file
        else:
            raise EvaluationError(
                f"Evaluation failed: No processed splits found for dataset '{config.dataset_id}'."
            )

        # Load samples lists
        try:
            raw_bytes = self.storage_manager.read_file(target_file)
            lines = raw_bytes.decode("utf-8").strip().split("\n")
            samples = [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: Failed to read dataset split: {e}") from e

        if not samples:
            raise EvaluationError(f"Evaluation failed: Dataset split '{target_file}' is empty.")

        # 3. Model Inference Execution
        evaluator = ModelEvaluator(config.model_id, checkpoint.path)
        predictions = evaluator.generate_predictions(
            samples=samples,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature
        )

        references = [sample.get("output") or "" for sample in samples]

        # 4. Compute Metrics
        metrics = MetricsEvaluator.evaluate(config.task_type, predictions, references)

        # 5. Compile structured report files
        report_dir = f"artifacts/reports/{config.evaluation_id}"
        self.storage_manager.ensure_dir(report_dir)
        
        report_data = {
            "evaluation_id": config.evaluation_id,
            "model_id": config.model_id,
            "dataset_id": config.dataset_id,
            "task_type": config.task_type.value,
            "metrics": metrics.model_dump(),
            "samples": [
                {
                    "instruction": s.get("instruction", ""),
                    "input": s.get("input", ""),
                    "reference": r,
                    "prediction": p
                }
                for s, r, p in zip(samples, references, predictions)
            ]
        }
        
        report_json_path = f"{report_dir}/report.json"
        self.storage_manager.write_file(
            report_json_path,
            json.dumps(report_data, indent=2)
        )

        # 6. Database registration updates
        runtime = time.time() - start_time
        result = EvaluationResult(
            evaluation_id=config.evaluation_id,
            model_id=config.model_id,
            dataset_id=config.dataset_id,
            task_type=config.task_type,
            metrics=metrics,
            runtime_seconds=round(runtime, 2),
            report_path=report_json_path,
            created_at=datetime.utcnow()
        )

        # Save to evaluation results repository
        self.model_registry.save_evaluation(result)

        # Update leaderboard index metrics registry
        # Determine primary metrics score to list
        primary_metric = "bleu" if config.task_type == TaskType.GENERATION else "accuracy"
        primary_value = getattr(metrics, primary_metric) or 0.0

        self.model_registry.save_benchmark_entry(
            benchmark_id=f"bench_{config.evaluation_id}",
            dataset_id=config.dataset_id,
            model_id=config.model_id,
            metric_name=primary_metric,
            metric_value=primary_value
        )

        return result
