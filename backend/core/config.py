"""LLMOps Studio Configuration Module.

Provides centralized configuration loading using Pydantic Settings.
Implements a cached settings factory (lru_cache) and supports environment-based
loading (.env, .env.development, .env.production) based on APP_ENV mapped to constants.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from backend.core.constants import AppEnv
from backend.core.exceptions import ConfigurationError
from backend.core.version import APP_NAME, VERSION

# Resolve workspace root dynamically relative to this module
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings managed through Pydantic and configuration files."""

    app_env: str = Field(
        default=AppEnv.DEVELOPMENT.value, 
        validation_alias="APP_ENV"
    )

    # API credentials configuration
    huggingface_api_key: Optional[str] = Field(
        default=None, 
        validation_alias="HUGGINGFACE_API_KEY"
    )
    openai_api_key: Optional[str] = Field(
        default=None, 
        validation_alias="OPENAI_API_KEY"
    )

    # Default foundation model
    default_model_name: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        validation_alias="DEFAULT_MODEL_NAME"
    )

    # Path settings
    workspace_root: Path = WORKSPACE_ROOT
    data_dir: Path = WORKSPACE_ROOT / "data"
    model_dir: Path = WORKSPACE_ROOT / "models"
    artifact_dir: Path = WORKSPACE_ROOT / "artifacts"
    log_dir: Path = WORKSPACE_ROOT / "logs"
    config_yaml_path: Path = WORKSPACE_ROOT / "configs" / "config.yaml"

    # Default app identifiers (from backend/core/version.py)
    app_name: str = APP_NAME
    app_version: str = VERSION

    # QLoRA Hyperparameter defaults (from config.yaml training schema)
    learning_rate: float = 0.0002
    batch_size: int = 4
    num_train_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj"]

    # Metrics defaults (from config.yaml evaluation schema)
    default_metrics: List[str] = [
        "rouge",
        "bertscore",
        "exact_match",
        "latency",
        "throughput",
        "llm_as_a_judge",
        "cost_estimation"
    ]

    # Hardware constraints settings (from config.yaml benchmarking schema)
    cuda_visible_devices: str = "0"
    precision: str = "fp16"
    max_tokens: int = 512

    # Immutability settings configuration
    model_config = SettingsConfigDict(
        extra="ignore",
        frozen=True
    )


def get_env_files(app_env: str) -> List[str]:
    """Determines which environment files to load based on the app environment stage.

    Args:
        app_env: Active environment name.

    Returns:
        List of dotenv file names.
    """
    env_files = [".env"]
    if app_env == AppEnv.DEVELOPMENT.value:
        env_files.append(".env.development")
    elif app_env == AppEnv.PRODUCTION.value:
        env_files.append(".env.production")
    return env_files


def load_yaml_overrides(yaml_path: Path) -> Dict[str, Any]:
    """Parses config.yaml and flattens properties into dictionary configurations.

    Args:
        yaml_path: Target path on disk.

    Returns:
        Dictionary mapping config keys to parameters.
    """
    overrides: Dict[str, Any] = {}
    if not yaml_path.exists():
        return overrides

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            
        if not yaml_data:
            return overrides

        # Apply App values
        app_cfg = yaml_data.get("app", {})
        if "name" in app_cfg:
            overrides["app_name"] = app_cfg["name"]
        if "version" in app_cfg:
            overrides["app_version"] = app_cfg["version"]

        # Apply Training Hyperparameters
        train_cfg = yaml_data.get("training", {})
        hparams = train_cfg.get("default_hyperparameters", {})
        for attr in [
            "learning_rate",
            "batch_size",
            "num_train_epochs",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules"
        ]:
            if attr in hparams:
                overrides[attr] = hparams[attr]

        # Apply Evaluation metrics
        eval_cfg = yaml_data.get("evaluation", {})
        if "default_metrics" in eval_cfg:
            overrides["default_metrics"] = eval_cfg["default_metrics"]

        # Apply Benchmarking hardware defaults
        bench_cfg = yaml_data.get("benchmarking", {})
        hw_cfg = bench_cfg.get("default_hardware", {})
        for attr in ["cuda_visible_devices", "precision", "max_tokens"]:
            if attr in hw_cfg:
                overrides[attr] = hw_cfg[attr]

        return overrides
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load yaml configurations from '{yaml_path}': {e}"
        ) from e


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached Settings configuration singleton.

    Loads env configurations (.env files, system env variables) and merges config.yaml values
    following priority rules: Defaults < YAML Config < Dotenv Files < System Environment Variables.
    """
    # 1. Query active APP_ENV stage
    app_env = os.getenv("APP_ENV", AppEnv.DEVELOPMENT.value).lower()
    env_files = get_env_files(app_env)

    # 2. Parse overrides from yaml configurations
    yaml_path = WORKSPACE_ROOT / "configs" / "config.yaml"
    yaml_values = load_yaml_overrides(yaml_path)

    # 3. Use a temporary instance to resolve DOTENV parameters priority
    temp_settings = Settings(_env_file=tuple(env_files))
    
    # 4. Resolve environment override values
    env_overrides = {
        key: getattr(temp_settings, key)
        for key in temp_settings.model_fields_set
    }

    # 5. Build final immutable Settings instance
    final_kwargs = {**yaml_values, **env_overrides}
    return Settings(**final_kwargs, _env_file=tuple(env_files))
