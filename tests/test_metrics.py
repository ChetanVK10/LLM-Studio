"""Unit tests for the metrics calculators.

Validates that unigram BLEU, LCS ROUGE-L, and classification macro-averages
Accuracy, Precision, Recall, and F1 calculations are correct.
"""

from backend.evaluation.metrics import BleuMetric, MetricsEvaluator, RougeLMetric


def test_bleu_metric() -> None:
    """Asserts unigram token overlap BLEU precision score logic."""
    metric = BleuMetric()
    pred = ["the cat sat on the mat"]
    ref = ["the cat sat on the mat"]
    assert metric.compute(pred, ref) == 1.0
    
    pred = ["sat cat"]
    ref = ["the cat sat"]
    assert metric.compute(pred, ref) == 1.0


def test_rouge_l_metric() -> None:
    """Asserts longest common subsequence ROUGE-L calculations."""
    metric = RougeLMetric()
    pred = ["the cat sat"]
    ref = ["the cat sat"]
    assert metric.compute(pred, ref) == 1.0
    
    pred = ["the sat"]
    ref = ["the cat sat"]
    # LCS length is 2 (the, sat). Ref length=3, Pred length=2
    # recall = 2/3, precision = 2/2 = 1.0
    # f1 = 2 * (2/3 * 1) / (2/3 + 1) = 4/3 / 5/3 = 0.8
    assert round(metric.compute(pred, ref), 2) == 0.80


def test_classification_stats() -> None:
    """Asserts multiclass macro-averaged classification evaluation logic."""
    predictions = ["positive", "negative", "positive"]
    references = ["positive", "positive", "negative"]
    
    accuracy, precision, recall, f1 = MetricsEvaluator._compute_classification_stats(
        predictions, 
        references
    )
    
    assert round(accuracy, 2) == 0.33
    assert precision == 0.25
    assert recall == 0.25
    assert f1 == 0.25
