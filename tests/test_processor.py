"""Unit tests for dataset processor orchestrator.

Validates that ingestion workflows transition datasets through lifecycle states,
writes Parquet/JSONL splits to processed folders, and raises ValidationError when files
contain invalid schemas.
"""

import json

import pytest

from backend.core.exceptions import DatasetValidationError
from backend.datasets.loader import DatasetLoader
from backend.datasets.preview import DatasetPreviewGenerator
from backend.datasets.processor import DatasetProcessor
from backend.datasets.repository import SQLiteDatasetRepository
from backend.datasets.splitter import DatasetSplitter
from backend.datasets.token_estimator import HeuristicTokenEstimator
from backend.datasets.validator import DatasetValidator
from backend.schemas.datasets import DatasetSplitConfig
from backend.storage.local_storage import LocalStorage


def test_processor_pipeline_success(tmp_path) -> None:
    """Asserts that valid JSON records lists ingest, validate, profile, split, and register as READY."""
    storage = LocalStorage(tmp_path)
    repo = SQLiteDatasetRepository(tmp_path / "datasets.db")

    loader = DatasetLoader(storage)
    validator = DatasetValidator()
    estimator = HeuristicTokenEstimator()
    preview = DatasetPreviewGenerator(estimator)
    splitter = DatasetSplitter()

    processor = DatasetProcessor(
        storage_manager=storage,
        repository=repo,
        loader=loader,
        validator=validator,
        preview_generator=preview,
        splitter=splitter
    )

    # 1. Prepare raw valid dataset contents
    raw_data = [
        {"instruction": "What is gravity?", "input": "", "output": "Gravity attracts things."},
        {"instruction": "Translate hello", "input": "", "output": "Hola"},
        {"instruction": "Multiply numbers", "input": "2*3", "output": "6"}
    ]
    raw_content = json.dumps(raw_data).encode("utf-8")

    split_config = DatasetSplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)

    # 2. Trigger pipeline
    meta = processor.process_dataset(
        dataset_name="test_clean_dataset",
        content=raw_content,
        file_extension=".json",
        split_config=split_config
    )

    # 3. Assert configurations matches
    assert meta.dataset_name == "test_clean_dataset"
    assert meta.lifecycle_state.value == "ready"
    assert meta.profile is not None
    assert meta.profile.rows_count == 3
    assert meta.profile.estimated_tokens > 0
    assert len(meta.profile.columns) == 3

    # 4. Check that processed partitions splits were written via storage
    assert storage.exists(f"data/processed/{meta.dataset_id}/train.jsonl")
    assert storage.exists(f"data/processed/{meta.dataset_id}/validation.jsonl")
    assert storage.exists(f"data/processed/{meta.dataset_id}/test.jsonl")

    # 5. Check database catalog update
    retrieved = repo.get_by_id(meta.dataset_id)
    assert retrieved is not None
    assert retrieved.lifecycle_state.value == "ready"


def test_processor_pipeline_validation_error(tmp_path) -> None:
    """Asserts that invalid dataset formats register in DB up to VALIDATED state and raise error."""
    storage = LocalStorage(tmp_path)
    repo = SQLiteDatasetRepository(tmp_path / "datasets.db")

    loader = DatasetLoader(storage)
    validator = DatasetValidator()
    estimator = HeuristicTokenEstimator()
    preview = DatasetPreviewGenerator(estimator)
    splitter = DatasetSplitter()

    processor = DatasetProcessor(
        storage_manager=storage,
        repository=repo,
        loader=loader,
        validator=validator,
        preview_generator=preview,
        splitter=splitter
    )

    # 1. Prepare raw malformed dataset contents (lacks instruction and output keys)
    raw_data = [{"invalid_header_key": "some cell content"}]
    raw_content = json.dumps(raw_data).encode("utf-8")

    # 2. Assert pipeline aborts with error
    with pytest.raises(DatasetValidationError):
        processor.process_dataset(
            dataset_name="malformed_set",
            content=raw_content,
            file_extension=".json"
        )

    # 3. Check that the db registry catalog holds logs showing state is VALIDATED (failed state)
    records = repo.list_all()
    assert len(records) == 1
    assert records[0].lifecycle_state.value == "validated"
    assert records[0].validation_report.is_valid is False
    assert len(records[0].validation_report.issues) > 0
