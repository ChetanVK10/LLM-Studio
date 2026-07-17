"""LLMOps Studio Filesystem Utilities.

Contains folder structure templates, directory initialization checks, path writability
validators, and the startup verification sequence.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

from backend.core.config import get_settings
from backend.core.constants import DirectoryName
from backend.core.exceptions import ConfigurationError, LLMOpsStudioError

logger = logging.getLogger("app")


def ensure_directories_exist() -> None:
    """Ensures all raw/processed data paths and artifact folders exist.

    Recursively creates missing directories on the filesystem.
    """
    settings = get_settings()
    required_dirs: List[Path] = [
        # Data storage directories
        settings.data_dir / DirectoryName.DATA_RAW.value,
        settings.data_dir / DirectoryName.DATA_PROCESSED.value,
        settings.data_dir / DirectoryName.DATA_CACHE.value,
        settings.data_dir / DirectoryName.DATA_UPLOADS.value,
        
        # Artifacts repositories
        settings.artifact_dir / DirectoryName.ARTIFACTS_CHECKPOINTS.value,
        settings.artifact_dir / DirectoryName.ARTIFACTS_EVALUATIONS.value,
        settings.artifact_dir / DirectoryName.ARTIFACTS_REPORTS.value,
        settings.artifact_dir / DirectoryName.ARTIFACTS_BENCHMARKS.value,
        settings.artifact_dir / DirectoryName.ARTIFACTS_EXPORTS.value,
        
        # System models directory
        settings.model_dir,
        
        # Logging logs directory
        settings.log_dir,
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def verify_directory_writability(path: Path) -> bool:
    """Validates that a path is writable by creating and deleting a temporary file.

    Args:
        path: Path representing target folder.

    Returns:
        True if the directory exists and allows write operations, False otherwise.
    """
    if not path.exists() or not path.is_dir():
        return False
    try:
        # Create a transient temporary file within the target folder to verify write permissions
        fd, temp_file_path = tempfile.mkstemp(dir=path)
        os.close(fd)
        os.remove(temp_file_path)
        return True
    except Exception:
        return False


def run_startup_validation() -> bool:
    """Orchestrates system startup checks.

    Performs:
      1. Settings verification.
      2. Missing directories initialization.
      3. Writability validations.
      4. Logger path checks.

    Returns:
        True if the system passes validation.

    Raises:
        ConfigurationError: If configs fail checks.
        LLMOpsStudioError: If folders are not writable.
    """
    settings = get_settings()
    try:
        # 1. Config paths check
        if not settings.workspace_root.exists():
            raise ConfigurationError(
                f"Workspace root directory does not exist: {settings.workspace_root}"
            )

        # 2. Automatically generate required folder structure
        ensure_directories_exist()

        # 3. Verify write permissions on active workspaces
        critical_paths: List[Path] = [
            settings.data_dir,
            settings.artifact_dir,
            settings.log_dir,
        ]
        for path in critical_paths:
            if not verify_directory_writability(path):
                raise LLMOpsStudioError(
                    f"Required system directory is not writable: {path}"
                )

        # 4. Perform log confirmation writing
        logger.info("Startup validation: Central logging system initialized.")
        logger.info(
            f"Startup validation: Configuration file loaded successfully. Environment: '{settings.app_env}'"
        )
        logger.info("Startup validation: System workspaces are writable and ready.")
        
        return True
    except Exception as e:
        # Standard fallback printing in case loggers are failing
        print(f"CRITICAL: LLMOps Studio Startup Validation Failed: {e}")
        if isinstance(e, LLMOpsStudioError):
            raise e
        raise LLMOpsStudioError(f"Startup validation failure: {e}") from e
