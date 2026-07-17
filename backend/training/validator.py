"""Pre-Training Validator Module.

Validates model identifier formats, dataset lifecycle readiness in registry,
output paths writability, and hyperparameter configuration ranges prior to launching jobs.
"""

import re
from pathlib import Path
from typing import Any

from backend.core.constants import DatasetLifecycleState
from backend.core.exceptions import DatasetValidationError, HyperparameterError, TrainingError
from backend.schemas.training import TrainingJobConfig
from backend.storage.storage_manager import StorageManager


class PreTrainingValidator:
    """Runs structural checks and hyperparameter bounds verification before training starts."""

    @staticmethod
    def validate_readiness(
        config: TrainingJobConfig,
        dataset_service: Any,
        storage_manager: StorageManager
    ) -> None:
        """Audits configurations parameters, target paths, and registered datasets.

        Args:
            config: Target training configuration properties.
            dataset_service: Registry client to check dataset states.
            storage_manager: Storage abstraction layer to verify paths.

        Raises:
            DatasetValidationError: If dataset is missing or not READY.
            HyperparameterError: If learning parameters are out of ranges.
            TrainingError: If target path or model identifiers are malformed.
        """
        # 1. Dataset existence and state checking
        dataset = dataset_service.get_dataset(config.dataset_id)
        if not dataset:
            raise DatasetValidationError(
                f"Pre-training validation failed: Dataset ID '{config.dataset_id}' "
                "is not registered in catalog."
            )
        
        if dataset.lifecycle_state != DatasetLifecycleState.READY:
            raise DatasetValidationError(
                f"Pre-training validation failed: Dataset '{dataset.dataset_name}' "
                f"is not ready for training (current state: '{dataset.lifecycle_state.value}')."
            )

        # 2. Model identifier verification
        model_name = config.base_model_name.strip()
        if not model_name:
            raise TrainingError("Pre-training validation failed: Base model name cannot be empty.")
            
        # Matches HuggingFace repository structures (e.g. 'org/repo') or local directory paths
        is_hf_repo = bool(re.match(r"^[\w\-.]+/[\w\-.]+$", model_name))
        is_local_path = Path(model_name).exists()
        is_generic_name = "/" not in model_name
        
        if not (is_hf_repo or is_local_path or is_generic_name):
            raise TrainingError(
                f"Pre-training validation failed: Base model name format '{model_name}' is invalid."
            )

        # 3. Output path check
        try:
            checkpoint_dir = f"artifacts/checkpoints/{config.job_id}"
            storage_manager.ensure_dir(checkpoint_dir)
        except Exception as e:
            raise TrainingError(
                f"Pre-training validation failed: Output checkpoint path is not writable: {e}"
            ) from e

        # 4. Hyperparameter validations
        hp = config.hyperparameters
        if hp.epochs < 1:
            raise HyperparameterError(
                f"Pre-training validation failed: Epochs count must be >= 1 (got {hp.epochs})."
            )
        if hp.batch_size < 1:
            raise HyperparameterError(
                f"Pre-training validation failed: Batch size must be >= 1 (got {hp.batch_size})."
            )
        if hp.learning_rate <= 0:
            raise HyperparameterError(
                f"Pre-training validation failed: Learning rate must be > 0 (got {hp.learning_rate})."
            )
        if not (0.0 <= hp.warmup_ratio <= 1.0):
            raise HyperparameterError(
                f"Pre-training validation failed: Warmup ratio must be in [0.0, 1.0] (got {hp.warmup_ratio})."
            )
        if hp.weight_decay < 0.0:
            raise HyperparameterError(
                f"Pre-training validation failed: Weight decay cannot be negative (got {hp.weight_decay})."
            )
        if hp.logging_steps < 1:
            raise HyperparameterError(
                f"Pre-training validation failed: Logging steps count must be >= 1 (got {hp.logging_steps})."
            )

        # 5. LoRA configuration validations
        lora = config.lora_config
        if lora.r < 1:
            raise HyperparameterError(
                f"Pre-training validation failed: LoRA rank dimension must be >= 1 (got {lora.r})."
            )
        if lora.lora_alpha < 1:
            raise HyperparameterError(
                f"Pre-training validation failed: LoRA alpha scaling must be >= 1 (got {lora.lora_alpha})."
            )
        if not (0.0 <= lora.lora_dropout <= 1.0):
            raise HyperparameterError(
                f"Pre-training validation failed: LoRA dropout must be in [0.0, 1.0] (got {lora.lora_dropout})."
            )
        if not lora.target_modules:
            raise HyperparameterError(
                "Pre-training validation failed: LoRA target modules list cannot be empty."
            )
