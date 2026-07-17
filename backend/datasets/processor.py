"""Dataset Ingestion Processor.

Orchestrates the sequence of loading, validating, profiling, splitting,
and registering raw datasets records, transitioning through lifecycle states.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import DatasetValidationError
from backend.datasets.hashing import calculate_content_hash
from backend.datasets.loader import DatasetLoader
from backend.datasets.preview import DatasetPreviewGenerator
from backend.datasets.repository import IDatasetRepository
from backend.datasets.splitter import DatasetSplitter
from backend.datasets.validator import DatasetValidator
from backend.schemas.datasets import DatasetMetadata, DatasetSplitConfig
from backend.storage.storage_manager import StorageManager

logger = logging.getLogger("app")


class DatasetProcessor:
    """Orchestrates processing pipelines and manages metadata transaction states."""

    def __init__(
        self,
        storage_manager: StorageManager,
        repository: IDatasetRepository,
        loader: DatasetLoader,
        validator: DatasetValidator,
        preview_generator: DatasetPreviewGenerator,
        splitter: DatasetSplitter
    ) -> None:
        """Initializes orchestrator.

        Args:
            storage_manager: Interface provider for file writing.
            repository: Metadata database registry manager.
            loader: Formatted files parsing utility.
            validator: Quality and schema check validator.
            preview_generator: Statistics profiler manager.
            splitter: Partition splits generator.
        """
        self.storage_manager = storage_manager
        self.repository = repository
        self.loader = loader
        self.validator = validator
        self.preview_generator = preview_generator
        self.splitter = splitter

    def process_dataset(
        self,
        dataset_name: str,
        content: bytes,
        file_extension: str,
        source: str = "local",
        split_config: Optional[DatasetSplitConfig] = None
    ) -> DatasetMetadata:
        """Runs validation and split workflows, exporting outcomes to processed partitions.

        Args:
            dataset_name: Logical display name.
            content: Raw file content bytes.
            file_extension: Parsing format type (e.g. '.csv', '.json', '.jsonl').
            source: Source details ('local' or 'huggingface').
            split_config: Configuration configurations for splitting.

        Returns:
            Saved DatasetMetadata details.

        Raises:
            DatasetValidationError: If validation checks fail.
        """
        if split_config is None:
            split_config = DatasetSplitConfig()

        ext = file_extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"

        # 1. Hashing and duplication verification
        content_hash = calculate_content_hash(content)
        logger.info(f"DatasetProcessor: Processing '{dataset_name}', SHA256: '{content_hash}'")

        # In case of existing matching READY entries, return cached metadata immediately
        existing = self.repository.get_by_id(content_hash)
        if existing and existing.lifecycle_state == DatasetLifecycleState.READY:
            logger.info("DatasetProcessor: Cache hit! Returning existing metadata record.")
            return existing

        # 2. Write raw content via storage abstraction layer
        raw_path = f"data/raw/{content_hash}{ext}"
        self.storage_manager.write_file(raw_path, content)

        # Initialize base metadata representation on disk (State: UPLOADED)
        metadata = DatasetMetadata(
            dataset_id=content_hash,
            dataset_name=dataset_name,
            raw_path=raw_path,
            lifecycle_state=DatasetLifecycleState.UPLOADED,
            source=source,
            version=1,
            created_at=datetime.utcnow(),
            split_config=split_config
        )
        self.repository.save(metadata)

        # 3. Load records
        records = self.loader.load_dataset(raw_path, ext)

        # 4. Perform validation (State: VALIDATED)
        validation_report = self.validator.validate_records(records)
        metadata.validation_report = validation_report
        metadata.lifecycle_state = DatasetLifecycleState.VALIDATED
        self.repository.save(metadata)

        if not validation_report.is_valid:
            logger.warning(f"DatasetProcessor: Validation check failed for '{dataset_name}'")
            raise DatasetValidationError("Dataset records failed validation checks.")

        # 5. Extract statistics profile (State: PROFILED)
        profile = self.preview_generator.generate_profile(records)
        metadata.profile = profile
        metadata.lifecycle_state = DatasetLifecycleState.PROFILED
        self.repository.save(metadata)

        # 6. Split data segments (State: SPLIT)
        splits = self.splitter.split_dataset(records, split_config)
        metadata.lifecycle_state = DatasetLifecycleState.SPLIT
        self.repository.save(metadata)

        # 7. Write splits to processed storage (State: REGISTERED)
        processed_dir = f"data/processed/{content_hash}"
        self.storage_manager.ensure_dir(processed_dir)

        for split_name, split_records in splits.items():
            split_path = f"{processed_dir}/{split_name}.jsonl"
            split_text = "\n".join(json.dumps(r) for r in split_records)
            self.storage_manager.write_file(split_path, split_text)

        metadata.processed_path = processed_dir
        metadata.lifecycle_state = DatasetLifecycleState.REGISTERED
        self.repository.save(metadata)

        # 8. Mark ready (State: READY)
        metadata.lifecycle_state = DatasetLifecycleState.READY
        self.repository.save(metadata)

        logger.info(f"DatasetProcessor: Processing completed successfully for '{dataset_name}'.")
        return metadata
