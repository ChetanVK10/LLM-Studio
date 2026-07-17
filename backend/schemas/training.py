"""Training Schemas.

Defines Pydantic structures for model training pipeline configurations,
hyperparameters, LoRA parameters, and job tracking statuses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingJobState(str, Enum):
    """Execution state representing the active status of a fine-tuning run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LoraConfigSchema(BaseModel):
    """Configuration parameters representing LoRA adapter settings."""

    r: int = Field(
        8, 
        ge=1, 
        description="LoRA rank dimension representing adaptor low-rank value"
    )
    lora_alpha: int = Field(
        16, 
        ge=1, 
        description="Scaling factor for LoRA weights mapping"
    )
    lora_dropout: float = Field(
        0.05, 
        ge=0.0, 
        le=1.0, 
        description="Dropout rate applied to LoRA adapters"
    )
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Model layer modules target vectors to apply LoRA matrices"
    )


class HyperparametersSchema(BaseModel):
    """Hyperparameters configuration controlling the fine-tuning optimization process."""

    learning_rate: float = Field(
        2e-4, 
        gt=0.0, 
        description="Initial learning rate for optimization steps"
    )
    batch_size: int = Field(
        4, 
        ge=1, 
        description="Training batch size per device"
    )
    epochs: int = Field(
        3, 
        ge=1, 
        description="Number of complete training passes over the dataset"
    )
    warmup_ratio: float = Field(
        0.03, 
        ge=0.0, 
        le=1.0, 
        description="Warmup steps ratio relative to total training steps"
    )
    weight_decay: float = Field(
        0.01, 
        ge=0.0, 
        description="Weight decay coefficient for optimization"
    )
    logging_steps: int = Field(
        10, 
        ge=1, 
        description="Number of steps between status logging updates"
    )


class TrainingJobConfig(BaseModel):
    """Complete configuration parameters defining a local model fine-tuning job."""

    job_id: str = Field(
        ..., 
        description="Unique identifier key representing this training job run"
    )
    base_model_name: str = Field(
        ..., 
        description="Identifier of the foundation model on Hugging Face or local path"
    )
    dataset_id: str = Field(
        ..., 
        description="Unique registry search key (SHA-256 hash) for the training dataset"
    )
    hyperparameters: HyperparametersSchema = Field(
        default_factory=HyperparametersSchema,
        description="Optimization configurations parameters"
    )
    lora_config: LoraConfigSchema = Field(
        default_factory=LoraConfigSchema,
        description="LoRA adapter target weights settings"
    )


class TrainingJobStatus(BaseModel):
    """Runtime stats and lifecycle details of an active or finished training job."""

    job_id: str = Field(
        ..., 
        description="Matches config job ID"
    )
    state: TrainingJobState = Field(
        default=TrainingJobState.PENDING,
        description="Current state in the job lifecycle"
    )
    current_epoch: float = Field(
        0.0, 
        description="Decimals indicating training epoch progress"
    )
    current_step: int = Field(
        0, 
        description="Total optimization steps completed"
    )
    total_steps: int = Field(
        0, 
        description="Target number of training steps calculated from splits"
    )
    train_loss: Optional[float] = Field(
        None, 
        description="Loss value at the most recent logging step"
    )
    eval_loss: Optional[float] = Field(
        None, 
        description="Validation loss computed during evaluation splits check"
    )
    elapsed_seconds: float = Field(
        0.0, 
        description="Total elapsed running time in seconds"
    )
    checkpoint_path: Optional[str] = Field(
        None, 
        description="Folder path where trained weights adapter outputs reside"
    )
    error_message: Optional[str] = Field(
        None, 
        description="Runtime exception message details if job failed"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Initial creation timestamp"
    )


class TrainingRequest(BaseModel):
    """Legacy compatibility placeholder trigger wrapper."""
    model_name: str
    dataset_name: str
    learning_rate: float = 0.0002
    batch_size: int = 4
    epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingResult(BaseModel):
    """Legacy compatibility placeholder completion wrapper."""
    run_id: str
    status: str
    output_checkpoint_dir: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
