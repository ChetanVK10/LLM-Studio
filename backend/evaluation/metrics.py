"""Metrics Engines and Calculations.

Defines object-oriented metric interfaces and calculations for generation evaluations
(BLEU, ROUGE-L, and mocked BERTScore) and classification evaluations (Accuracy, Macro-Precision,
Macro-Recall, and Macro-F1).
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List

from backend.schemas.evaluation import MetricSummary, TaskType


class BaseMetric(ABC):
    """Abstract Base Class defining evaluation metrics structures."""

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Computes metric score over lists of prediction and reference text values.

        Args:
            predictions: Model generated outputs.
            references: Target ground truth answers.

        Returns:
            Computed float score.
        """
        pass


class BleuMetric(BaseMetric):
    """Computes token-overlap unigram BLEU-1 precision score."""

    def compute(self, predictions: List[str], references: List[str]) -> float:
        if not predictions or not references:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            pred_counts = Counter(pred_tokens)
            ref_counts = Counter(ref_tokens)
            
            overlaps = sum(min(count, ref_counts.get(t, 0)) for t, count in pred_counts.items())
            scores.append(overlaps / len(pred_tokens))

        return sum(scores) / len(scores)


class RougeLMetric(BaseMetric):
    """Computes ROUGE-L score based on Longest Common Subsequence (LCS)."""

    def compute(self, predictions: List[str], references: List[str]) -> float:
        if not predictions or not references:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            lcs_len = self._lcs_length(pred_tokens, ref_tokens)
            
            # Recall and Precision calculations
            recall = lcs_len / len(ref_tokens)
            precision = lcs_len / len(pred_tokens)
            
            if recall + precision > 0:
                f_score = (2 * recall * precision) / (recall + precision)
            else:
                f_score = 0.0
            scores.append(f_score)

        return sum(scores) / len(scores)

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Computes Longest Common Subsequence length using dynamic programming."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    
        return dp[m][n]


class BertScoreMetric(BaseMetric):
    """Placeholder interface simulating BERTScore semantic evaluations."""

    def compute(self, predictions: List[str], references: List[str]) -> float:
        # Mock semantic alignment score
        return 0.85


class MetricsEvaluator:
    """Orchestrator matching task enums with target metric evaluation classes."""

    @staticmethod
    def evaluate(
        task_type: TaskType,
        predictions: List[str],
        references: List[str]
    ) -> MetricSummary:
        """Calculates metric statistics.

        Args:
            task_type: Target category (generation or classification).
            predictions: List of model prediction strings.
            references: Ground truth target strings.

        Returns:
            MetricSummary containing computed evaluations scores.
        """
        if task_type == TaskType.GENERATION:
            bleu = BleuMetric().compute(predictions, references)
            rouge = RougeLMetric().compute(predictions, references)
            bert = BertScoreMetric().compute(predictions, references)
            
            return MetricSummary(
                bleu=round(bleu, 4),
                rouge_l=round(rouge, 4),
                bert_score=round(bert, 4)
            )
        
        else:
            # Classification macro evaluation
            accuracy, precision, recall, f1 = MetricsEvaluator._compute_classification_stats(
                predictions, 
                references
            )
            return MetricSummary(
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1_score=round(f1, 4)
            )

    @staticmethod
    def _compute_classification_stats(
        predictions: List[str],
        references: List[str]
    ) -> tuple:
        """Macro-averaged statistics for classification outputs."""
        if not predictions or not references:
            return 0.0, 0.0, 0.0, 0.0

        # Exact matching accuracy ratio
        correct = sum(
            1 for p, r in zip(predictions, references) 
            if p.strip().lower() == r.strip().lower()
        )
        accuracy = correct / len(predictions)

        # Retrieve unique classes from references
        unique_classes = set(r.strip().lower() for r in references)
        if not unique_classes:
            return accuracy, 0.0, 0.0, 0.0

        precisions = []
        recalls = []

        for c in unique_classes:
            tp = sum(
                1 for p, r in zip(predictions, references) 
                if p.strip().lower() == c and r.strip().lower() == c
            )
            fp = sum(
                1 for p, r in zip(predictions, references) 
                if p.strip().lower() == c and r.strip().lower() != c
            )
            fn = sum(
                1 for p, r in zip(predictions, references) 
                if p.strip().lower() != c and r.strip().lower() == c
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)

        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)

        if macro_precision + macro_recall > 0:
            macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
        else:
            macro_f1 = 0.0

        return accuracy, macro_precision, macro_recall, macro_f1
