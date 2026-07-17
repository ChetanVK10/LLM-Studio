"""Local Storage Integration Module.

Placeholder module outlining future responsibilities for read/write checkpoint files,
saving evaluation json dumps, and flushing system cache folders.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("app")


class LocalStorageIntegration:
    """Manages raw dataset files, reports, and weight assets stored in local directory folders."""

    def __init__(self, workspace_root: Path) -> None:
        """Initializes local storage disk helper.

        Args:
            workspace_root: Active project directory.
        """
        self.workspace_root = workspace_root
        logger.info(f"Initialized Local Storage Integration adapter. Root: '{workspace_root}'")

    def save_json_artifact(self, relative_path: str, data_content: str) -> str:
        """Saves structured data outputs into relative paths under artifacts.

        Args:
            relative_path: Target filename (e.g. 'evaluations/eval_001.json').
            data_content: Serialized string content to write.

        Returns:
            The resolved absolute target filepath string.
        """
        logger.info(f"Local Storage: Mocking write file to 'artifacts/{relative_path}'")
        return str(self.workspace_root / "artifacts" / relative_path)

    def scan_checkpoints(self) -> List[str]:
        """Scans the adapter checkpoint directories and returns sub-folder path names.

        Returns:
            List of directory path strings.
        """
        logger.info("Local Storage: Scanning checkpoints directories.")
        return ["qlora_llama_001"]
