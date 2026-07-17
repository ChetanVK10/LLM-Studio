"""Training Schemas.

Defines Pydantic structures for model training pipeline triggers and summaries.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Configuration parameters requested to trigger a model fine-tuning run."""
    model_name: str = Field(
        ..., 
        description="Name or identifier of the foundation model to fine-tune"
    )
    dataset_name: str = Field(
        ..., 
        description="Name of the registered training dataset"
    )
    learning_rate: float = Field(
        0.0002, 
        description="Learning rate for optimization steps"
    )
    batch_size: int = Field(
        4, 
        description="Training batch size per device"
    )
    epochs: int = Field(
        3, 
        description="Number of complete training passes over the dataset"
    )
    lora_r: int = Field(
        16, 
        description="LoRA rank dimension representing adaptor matrix low-rank value"
    )
    lora_alpha: int = Field(
        32, 
        description="Scaling factor for LoRA weights mapping"
    )
    lora_dropout: float = Field(
        0.05, 
        description="Dropout rate applied to LoRA adapters"
    )
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Model layer modules target vectors to apply LoRA matrices"
    )


class TrainingResult(BaseModel):
    """Summarized outputs generated after completing an LLM training run."""
    run_id: str = Field(
        ..., 
        description="Unique experiment tracker run identifier"
    )
    status: str = Field(
        ..., 
        description="Lifecycle completion status of the training pipeline"
    )
    output_checkpoint_dir: Optional[str] = Field(
        None, 
        description="Physical filepath location storing final adapter weights"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Extracted key performance logs (e.g. final loss values)"
    )
