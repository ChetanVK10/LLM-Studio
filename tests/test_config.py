"""Unit tests for settings configurations, version properties, and system health checks.

Validates that environment variables parse correctly, version constants are mapped,
and health checks return healthy status configurations.
"""

import pytest
from backend.core.config import Settings, get_settings
from backend.core.version import APP_NAME, RELEASE_STAGE, VERSION
from backend.utils.health import check_system_health


def test_settings_default_values() -> None:
    """Asserts default parameters match config.yaml schema overrides."""
    custom_settings = Settings()
    
    # Defaults from config.yaml overrides
    assert custom_settings.app_name == "LLMOps Studio"
    assert custom_settings.learning_rate == 0.0002
    assert custom_settings.batch_size == 4
    assert custom_settings.lora_r == 16
    assert "rouge" in custom_settings.default_metrics


def test_settings_env_overrides(monkeypatch) -> None:
    """Verifies that environment overrides dynamically update class fields."""
    monkeypatch.setenv("DEFAULT_MODEL_NAME", "custom/test-model")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_secret")
    
    # Reload settings with monkeypatched env
    custom_settings = Settings()
    
    assert custom_settings.default_model_name == "custom/test-model"
    assert custom_settings.huggingface_api_key == "hf_test_secret"


def test_get_settings_cache() -> None:
    """Verifies that get_settings factory caches the Settings instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_version_constants() -> None:
    """Asserts version.py contains accurate release configurations."""
    assert APP_NAME == "LLMOps Studio"
    assert VERSION == "0.1.0"
    assert RELEASE_STAGE == "alpha"


def test_system_health_check() -> None:
    """Verifies that health monitor checks return correct statuses."""
    health = check_system_health()
    assert health["status"] == "healthy"
    assert health["version"] == VERSION
    assert health["checks"]["configuration"] == "pass"
