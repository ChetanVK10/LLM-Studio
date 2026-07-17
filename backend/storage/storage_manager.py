"""Storage Manager Abstraction.

Defines the StorageManager abstract base class to decouple the backend code from
physical filesystem APIs, facilitating future cloud storage migrations (S3, GCS).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class StorageManager(ABC):
    """Abstract interface exposing file storage operations."""

    @abstractmethod
    def read_file(self, path: Union[str, Path]) -> bytes:
        """Reads binary file contents and returns bytes.

        Args:
            path: Target file path or identifier.

        Returns:
            The raw bytes of the file.
        """
        pass

    @abstractmethod
    def write_file(self, path: Union[str, Path], content: Union[bytes, str]) -> str:
        """Writes content to target path.

        Args:
            path: Target file path or identifier.
            content: String or bytes content to write.

        Returns:
            The resolved target path string.
        """
        pass

    @abstractmethod
    def delete_file(self, path: Union[str, Path]) -> bool:
        """Deletes file at target path.

        Args:
            path: Target file path or identifier.

        Returns:
            True if the file was deleted, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Checks if a file exists at target path.

        Args:
            path: Target path or identifier.

        Returns:
            True if file exists, False otherwise.
        """
        pass

    @abstractmethod
    def ensure_dir(self, path: Union[str, Path]) -> None:
        """Ensures that the directory structure exists recursively.

        Args:
            path: Target directory path.
        """
        pass
