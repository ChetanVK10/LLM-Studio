"""Dataset Service Layer.

Orchestrates local uploads, Hugging Face downloads, statistics profiling,
and database query retrieval.
"""

import logging
from typing import List, Optional

from backend.datasets.processor import DatasetProcessor
from backend.datasets.repository import IDatasetRepository
from backend.integrations.huggingface import HuggingFaceDatasetIngestor
from backend.schemas.datasets import DatasetMetadata, DatasetSplitConfig

logger = logging.getLogger("app")


class DatasetService:
    """Orchestrates filesystem data pipelines and splits registries."""

    def __init__(
        self,
        processor: DatasetProcessor,
        repository: IDatasetRepository,
        hf_ingestor: HuggingFaceDatasetIngestor
    ) -> None:
        """Initializes the dataset service.

        Args:
            processor: Ingestion pipeline orchestrator.
            repository: Metadata persistence manager.
            hf_ingestor: Hugging Face download adapter.
        """
        self.processor = processor
        self.repository = repository
        self.hf_ingestor = hf_ingestor

    def import_local_dataset(
        self,
        name: str,
        content: bytes,
        file_extension: str,
        split_config: Optional[DatasetSplitConfig] = None
    ) -> DatasetMetadata:
        """Processes and registers a local file upload.

        Args:
            name: Display name.
            content: Raw binary file contents.
            file_extension: Extension suffix (e.g. '.json', '.csv').
            split_config: Split parameters configuration.

        Returns:
            Saved DatasetMetadata.
        """
        logger.info(f"DatasetService: Ingesting local file '{name}' ({file_extension})")
        return self.processor.process_dataset(
            dataset_name=name,
            content=content,
            file_extension=file_extension,
            source="local",
            split_config=split_config
        )

    def import_huggingface_dataset(
        self,
        repo_id: str,
        split: str = "train",
        split_config: Optional[DatasetSplitConfig] = None
    ) -> DatasetMetadata:
        """Downloads, processes, and registers a Hugging Face Hub dataset.

        Args:
            repo_id: Repository ID identifier (e.g. 'tatsu-lab/alpaca').
            split: Segment category split (e.g. 'train').
            split_config: Split parameters configuration.

        Returns:
            Saved DatasetMetadata.
        """
        logger.info(f"DatasetService: Downloading HF repository '{repo_id}' (split: '{split}')")
        raw_bytes = self.hf_ingestor.fetch_dataset(repo_id, split)
        
        # Hugging Face imports return JSON data, so extension is '.json'
        dataset_name = repo_id.replace("/", "_")
        return self.processor.process_dataset(
            dataset_name=dataset_name,
            content=raw_bytes,
            file_extension=".json",
            source=f"huggingface:{repo_id}",
            split_config=split_config
        )

    def list_datasets(self) -> List[DatasetMetadata]:
        """Queries stored metadata index lists of all registered datasets.

        Returns:
            List of DatasetMetadata.
        """
        logger.info("DatasetService: Querying all cataloged dataset entries.")
        return self.repository.list_all()

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieves a single dataset's metadata.

        Args:
            dataset_id: Cryptographic content hash (SHA-256).

        Returns:
            DatasetMetadata if found, None otherwise.
        """
        logger.info(f"DatasetService: Retrieving metadata for '{dataset_id}'")
        return self.repository.get_by_id(dataset_id)

    def remove_dataset(self, dataset_id: str) -> bool:
        """Deletes a dataset index entry from database registry.

        Args:
            dataset_id: Cryptographic content hash.

        Returns:
            True if removed, False otherwise.
        """
        logger.info(f"DatasetService: Removing dataset register target '{dataset_id}'")
        return self.repository.delete(dataset_id)
