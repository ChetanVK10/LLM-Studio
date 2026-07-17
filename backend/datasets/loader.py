"""Dataset File Loader.

Reads and parses dataset records from local storage paths, supporting CSV,
JSON, and JSONL format parsers using standard Python parsing modules.
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from backend.core.exceptions import DatasetLoaderError
from backend.storage.storage_manager import StorageManager


class DatasetLoader:
    """Parses dataset records in CSV, JSON, and JSONL formats using the storage layer."""

    def __init__(self, storage_manager: StorageManager) -> None:
        """Initializes the loader.

        Args:
            storage_manager: Active storage abstraction provider.
        """
        self.storage_manager = storage_manager

    def load_dataset(self, path: Union[str, Path], file_extension: str) -> List[Dict[str, Any]]:
        """Reads file contents from storage and parses records based on file extension.

        Args:
            path: Target storage path.
            file_extension: Format extension (e.g. '.csv', '.json', '.jsonl').

        Returns:
            Parsed list of dictionaries representing record rows.

        Raises:
            DatasetLoaderError: If loading or parsing failures occur.
        """
        # 1. Retrieve raw bytes and decode
        try:
            raw_bytes = self.storage_manager.read_file(path)
            text_content = raw_bytes.decode("utf-8")
        except Exception as e:
            raise DatasetLoaderError(
                f"Failed to read dataset bytes from target path '{path}': {e}"
            ) from e

        ext = file_extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"

        records: List[Dict[str, Any]] = []

        # 2. Parse file content dynamically
        try:
            if ext == ".json":
                data = json.loads(text_content)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = [data]
                else:
                    raise DatasetLoaderError(
                        "JSON dataset file structure must represent a list or a single object dictionary."
                    )

            elif ext == ".jsonl":
                lines = text_content.splitlines()
                for idx, line in enumerate(lines):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        records.append(json.loads(line_str))
                    except json.JSONDecodeError as err:
                        raise DatasetLoaderError(
                            f"Malformed JSONL format at line {idx + 1}: {err}"
                        )

            elif ext == ".csv":
                csv_file = io.StringIO(text_content)
                reader = csv.DictReader(csv_file)
                records = [dict(row) for row in reader]

            else:
                raise DatasetLoaderError(
                    f"Unsupported dataset format file extension: '{file_extension}'"
                )

            return records

        except DatasetLoaderError:
            raise
        except Exception as e:
            raise DatasetLoaderError(f"Incompatible file contents formatting: {e}") from e
