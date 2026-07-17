"""Unit tests for dataset splitters.

Validates that splits ratio partitions sum to 100% correctly, random seeds
yield reproducible groupings, and invalid ratios trigger custom exceptions.
"""

import pytest

from backend.core.exceptions import InvalidSplitRatioError
from backend.datasets.splitter import DatasetSplitter
from backend.schemas.datasets import DatasetSplitConfig


def test_splitter_ratios() -> None:
    """Asserts that division indices match configuration ratio percentages."""
    splitter = DatasetSplitter()
    records = [{"instruction": f"Inst {i}", "output": f"Out {i}"} for i in range(10)]
    config = DatasetSplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42)
    
    splits = splitter.split_dataset(records, config)
    assert len(splits["train"]) == 8
    assert len(splits["validation"]) == 1
    assert len(splits["test"]) == 1
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 10


def test_splitter_reproducibility() -> None:
    """Asserts that identical seeds shuffle and split datasets identically."""
    splitter = DatasetSplitter()
    records = [{"instruction": f"Inst {i}", "output": f"Out {i}"} for i in range(100)]
    config = DatasetSplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=999)
    
    splits1 = splitter.split_dataset(records, config)
    splits2 = splitter.split_dataset(records, config)
    
    assert splits1["train"] == splits2["train"]
    assert splits1["validation"] == splits2["validation"]
    assert splits1["test"] == splits2["test"]


def test_splitter_invalid_ratios() -> None:
    """Asserts that ratio sums not matching 1.0 raise InvalidSplitRatioError."""
    splitter = DatasetSplitter()
    records = [{"instruction": "Inst", "output": "Out"}]
    # Sum is 1.2, which is out of bounds
    config = DatasetSplitConfig(train_ratio=0.9, val_ratio=0.2, test_ratio=0.1)
    
    with pytest.raises(InvalidSplitRatioError):
        splitter.split_dataset(records, config)
