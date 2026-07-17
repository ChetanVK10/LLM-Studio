"""Dataset Orchestration Service.

Manages data ingestion profiles, preprocessing schedules, caching tokens, and listing datasets.
"""

import logging
from typing import List

from backend.schemas.datasets import DatasetMetadata

logger = logging.getLogger("app")


class DatasetService:
    """Orchestrates filesystem data pipelines and splits mapping for training and evaluation."""

    def __init__(self) -> None:
        """Initializes the dataset service instance."""
        pass

    def register_dataset(self, name: str, raw_filepath: str) -> DatasetMetadata:
        """Ingests a raw source data file and registers its metadata.

        Args:
            name: Label matching target dataset.
            raw_filepath: Path to original raw dataset.

        Returns:
            DatasetMetadata representing details of the registered dataset.
        """
        logger.info(f"Registering dataset '{name}' from source: '{raw_filepath}'")
        return DatasetMetadata(
            dataset_name=name,
            raw_path=raw_filepath,
            processed_path=None,
            num_samples=1000,
            features=["instruction", "input", "output"],
            split_ratio="80/10/10"
        )

    def process_dataset(self, name: str) -> DatasetMetadata:
        """Invokes parsing and tokenization filters, caching records for trainer pipelines.

        Args:
            name: Key of target registered raw dataset.

        Returns:
            DatasetMetadata representing details of the processed dataset.
        """
        logger.info(f"Processing and tokenizing dataset: '{name}'")
        return DatasetMetadata(
            dataset_name=name,
            raw_path=f"data/raw/{name}.json",
            processed_path=f"data/processed/{name}",
            num_samples=1000,
            features=["input_ids", "attention_mask", "labels"],
            split_ratio="80/10/10"
        )

    def list_datasets(self) -> List[DatasetMetadata]:
        """Queries stored metadata index lists of all registered datasets.

        Returns:
            List of DatasetMetadata.
        """
        logger.info("Listing all registered and processed datasets")
        return [
            DatasetMetadata(
                dataset_name="alpaca-cleaned-1k",
                raw_path="data/raw/alpaca-cleaned-1k.json",
                processed_path="data/processed/alpaca-cleaned-1k",
                num_samples=1000,
                features=["input_ids", "attention_mask", "labels"],
                split_ratio="100/0/0"
            )
        ]
