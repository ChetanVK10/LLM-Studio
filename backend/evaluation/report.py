"""Evaluation Report Generator.

Formats evaluation run configurations, metrics, and sample predictions
into structured Markdown reports prepared for future export capabilities.
"""

from typing import Any, Dict, List

from backend.schemas.evaluation import EvaluationResult


class ReportGenerator:
    """Formats evaluation results and metrics into structured Markdown summary reports."""

    @staticmethod
    def generate_markdown_report(result: EvaluationResult, samples: List[Dict[str, Any]]) -> str:
        """Generates a detailed evaluation summary in Markdown format.

        Args:
            result: EvaluationResult summary metadata.
            samples: List of evaluated row predictions.

        Returns:
            Formatted Markdown report string.
        """
        md = [
            f"# Model Evaluation Report: {result.evaluation_id}",
            "",
            "## 1. Run Configuration",
            f"- **Evaluated Model ID**: `{result.model_id}`",
            f"- **Target Dataset ID**: `{result.dataset_id}`",
            f"- **Task Category**: `{result.task_type.value}`",
            f"- **Total Runtime**: {result.runtime_seconds} seconds",
            f"- **Evaluated Timestamp**: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 2. Evaluation Metrics Summary"
        ]

        metrics_dict = result.metrics.model_dump()
        for metric, value in metrics_dict.items():
            if value is not None:
                md.append(f"- **{metric.upper()}**: `{value:.4f}`")

        md.append("")
        md.append("## 3. Sample Inference Predictions Preview")
        md.append("| Instruction | Expected Ground Truth | Model Generation Output |")
        md.append("| :--- | :--- | :--- |")

        # Preview up to first 10 samples
        for s in samples[:10]:
            inst = s.get("instruction", "").replace("\n", " ")
            ref = s.get("reference", "").replace("\n", " ")
            pred = s.get("prediction", "").replace("\n", " ")
            
            # Truncate text for grid alignment
            inst_trunc = inst[:50] + "..." if len(inst) > 50 else inst
            ref_trunc = ref[:50] + "..." if len(ref) > 50 else ref
            pred_trunc = pred[:50] + "..." if len(pred) > 50 else pred
            
            md.append(f"| {inst_trunc} | {ref_trunc} | {pred_trunc} |")

        return "\n".join(md)
