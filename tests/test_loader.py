"""Unit tests for dataset file loaders.

Validates that CSV, JSON, and JSONL formats parse correctly and throw custom errors
when files are malformed or unsupported.
"""

import pytest

from backend.core.exceptions import DatasetLoaderError
from backend.datasets.loader import DatasetLoader
from backend.storage.local_storage import LocalStorage


def test_loader_csv(tmp_path) -> None:
    """Asserts that CSV files parse correct headers and records list."""
    storage = LocalStorage(tmp_path)
    loader = DatasetLoader(storage)
    
    csv_content = "instruction,input,output\nExplain gravity,,Gravity attracts things\nTranslate,,Hola\n"
    storage.write_file("data.csv", csv_content)
    
    records = loader.load_dataset("data.csv", ".csv")
    assert len(records) == 2
    assert records[0]["instruction"] == "Explain gravity"
    assert records[0]["input"] == ""
    assert records[1]["output"] == "Hola"


def test_loader_json(tmp_path) -> None:
    """Asserts that JSON list structures parse correct records."""
    storage = LocalStorage(tmp_path)
    loader = DatasetLoader(storage)
    
    json_content = '[{"instruction": "Explain gravity", "output": "Gravity attracts"}]'
    storage.write_file("data.json", json_content)
    
    records = loader.load_dataset("data.json", ".json")
    assert len(records) == 1
    assert records[0]["instruction"] == "Explain gravity"


def test_loader_jsonl(tmp_path) -> None:
    """Asserts that line-separated JSON files parse row-by-row successfully."""
    storage = LocalStorage(tmp_path)
    loader = DatasetLoader(storage)
    
    jsonl_content = '{"instruction": "Explain gravity", "output": "Gravity"}\n{"instruction": "Translate", "output": "Hola"}\n'
    storage.write_file("data.jsonl", jsonl_content)
    
    records = loader.load_dataset("data.jsonl", ".jsonl")
    assert len(records) == 2
    assert records[0]["instruction"] == "Explain gravity"
    assert records[1]["output"] == "Hola"


def test_loader_invalid_ext(tmp_path) -> None:
    """Asserts that unsupported extensions trigger DatasetLoaderError."""
    storage = LocalStorage(tmp_path)
    loader = DatasetLoader(storage)
    storage.write_file("data.txt", "unsupported text format")
    
    with pytest.raises(DatasetLoaderError):
        loader.load_dataset("data.txt", ".txt")
