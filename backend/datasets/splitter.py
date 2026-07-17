"""Dataset Splitter Module.

Slices dataset collections into train, validation, and test datasets based on Pydantic
ratios configs and ensures reproducibility by using local random seeds.
"""

import random
from typing import Any, Dict, List

from backend.core.exceptions import InvalidSplitRatioError
from backend.schemas.datasets import DatasetSplitConfig


class DatasetSplitter:
    """Splits dataset records into partitioned lists using local Random states."""

    def split_dataset(
        self, records: List[Dict[str, Any]], config: DatasetSplitConfig
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Splits master dataset list into train, validation, and test segments.

        Args:
            records: Complete list of record rows.
            config: Target ratio settings (train, validation, test) and random seeds.

        Returns:
            Dictionary mapping splits 'train', 'validation', and 'test' to record lists.

        Raises:
            InvalidSplitRatioError: If ratio checks fail.
        """
        # 1. Validate ratio boundaries and sums
        ratios = [config.train_ratio, config.val_ratio, config.test_ratio]
        
        if any(r < 0.0 or r > 1.0 for r in ratios):
            raise InvalidSplitRatioError("Individual split ratios must be within [0.0, 1.0].")

        ratio_sum = sum(ratios)
        if not (0.999 <= ratio_sum <= 1.001):
            raise InvalidSplitRatioError(
                f"Split ratios must sum to exactly 1.0. Got: {ratio_sum:.4f} "
                f"(train={config.train_ratio}, validation={config.val_ratio}, test={config.test_ratio})"
            )

        if not records:
            return {"train": [], "validation": [], "test": []}

        # 2. Shuffle records reproducibly using local RNG to avoid side effects
        shuffled = list(records)
        local_rng = random.Random(config.random_seed)
        local_rng.shuffle(shuffled)

        # 3. Calculate partition offsets
        total_count = len(shuffled)
        train_count = int(round(config.train_ratio * total_count))
        val_count = int(round(config.val_ratio * total_count))

        # Adjust partitions boundary indices
        train_end = train_count
        val_end = min(total_count, train_end + val_count)

        train_split = shuffled[:train_end]
        validation_split = shuffled[train_end:val_end]
        test_split = shuffled[val_end:]

        return {
            "train": train_split,
            "validation": validation_split,
            "test": test_split
        }
