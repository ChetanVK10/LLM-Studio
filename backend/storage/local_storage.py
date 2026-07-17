"""Local Disk Storage Implementation.

Implements the StorageManager interface targeting the local workspace directory structure.
"""

import logging
import os
from pathlib import Path
from typing import Union

from backend.storage.storage_manager import StorageManager

logger = logging.getLogger("app")


class LocalStorage(StorageManager):
    """Local file system implementation of the StorageManager abstraction."""

    def __init__(self, workspace_root: Path) -> None:
        """Initializes the local storage helper.

        Args:
            workspace_root: Absolute Path to the workspace directory.
        """
        self.workspace_root = workspace_root

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Helper to resolve relative paths against workspace root.

        Args:
            path: Target file path or identifier.

        Returns:
            Resolved absolute Path.
        """
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.workspace_root / p).resolve()

    def read_file(self, path: Union[str, Path]) -> bytes:
        """Reads file content as binary.

        Args:
            path: Path to target file.

        Returns:
            Raw bytes.
        """
        resolved = self._resolve_path(path)
        logger.info(f"LocalStorage: Reading file: {resolved}")
        with open(resolved, "rb") as f:
            return f.read()

    def write_file(self, path: Union[str, Path], content: Union[bytes, str]) -> str:
        """Writes text or binary content to disk path.

        Args:
            path: Path to target file.
            content: Bytes or string to write.

        Returns:
            Absolute target file path string.
        """
        resolved = self._resolve_path(path)
        self.ensure_dir(resolved.parent)
        logger.info(f"LocalStorage: Writing file: {resolved}")
        
        if isinstance(content, str):
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(resolved, "wb") as f:
                f.write(content)
                
        return str(resolved)

    def delete_file(self, path: Union[str, Path]) -> bool:
        """Deletes file from disk if it exists.

        Args:
            path: Path to target file.

        Returns:
            True if file existed and was deleted, False otherwise.
        """
        resolved = self._resolve_path(path)
        if resolved.exists() and resolved.is_file():
            logger.info(f"LocalStorage: Deleting file: {resolved}")
            os.remove(resolved)
            return True
        return False

    def exists(self, path: Union[str, Path]) -> bool:
        """Checks file existence on disk.

        Args:
            path: Path to check.

        Returns:
            True if exists, False otherwise.
        """
        resolved = self._resolve_path(path)
        return resolved.exists()

    def ensure_dir(self, path: Union[str, Path]) -> None:
        """Ensures target directory exists on disk.

        Args:
            path: Target directory path.
        """
        resolved = self._resolve_path(path)
        resolved.mkdir(parents=True, exist_ok=True)
