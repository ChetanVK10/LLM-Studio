"""Unit tests for filesystem utilities.

Validates that directory creation occurs recursively and path permissions checks function as intended.
"""

from backend.core.config import get_settings
from backend.utils.fs import ensure_directories_exist, verify_directory_writability, run_startup_validation


def test_ensure_directories_exist(tmp_path, monkeypatch) -> None:
    """Verifies that all required folders are created on load."""
    custom_settings = get_settings().model_copy(update={
        "data_dir": tmp_path / "data",
        "artifact_dir": tmp_path / "artifacts",
        "log_dir": tmp_path / "logs",
        "model_dir": tmp_path / "models"
    })
    monkeypatch.setattr("backend.utils.fs.get_settings", lambda: custom_settings)
    
    ensure_directories_exist()
    
    # Check specific partitions
    assert (custom_settings.data_dir / "raw").exists()
    assert (custom_settings.data_dir / "processed").exists()
    assert (custom_settings.data_dir / "cache").exists()
    assert (custom_settings.data_dir / "uploads").exists()
    assert (custom_settings.artifact_dir / "checkpoints").exists()
    assert (custom_settings.artifact_dir / "evaluations").exists()
    assert (custom_settings.artifact_dir / "reports").exists()
    assert (custom_settings.artifact_dir / "benchmarks").exists()
    assert (custom_settings.artifact_dir / "exports").exists()
    assert custom_settings.log_dir.exists()
    assert custom_settings.model_dir.exists()


def test_verify_directory_writability(tmp_path) -> None:
    """Checks directory validation returns accurate boolean flags."""
    assert verify_directory_writability(tmp_path) is True
    
    # Path that doesn't exist
    assert verify_directory_writability(tmp_path / "absent_dir") is False


def test_run_startup_validation() -> None:
    """Validates that system checks terminate with True in test configurations."""
    # Since workspace_root is valid in test contexts, run_startup_validation should pass
    assert run_startup_validation() is True
