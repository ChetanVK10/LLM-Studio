"""Registry Schemas.

Defines Pydantic structures for model checkpoint registration catalog entries.
"""

from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RegistryEntry(BaseModel):
    """Catalog index containing model adapter checkpoints metadata and location details."""
    checkpoint_id: str = Field(
        ..., 
        description="Unique cryptographic token or label ID for the checkpoint"
    )
    model_name: str = Field(
        ..., 
        description="Logical name cataloged for search query matching"
    )
    base_model: str = Field(
        ..., 
        description="Root pre-trained weights foundation model catalog source name"
    )
    path: str = Field(
        ..., 
        description="File path where weights files (e.g. safetensors) reside"
    )
    tags: List[str] = Field(
        default_factory=list, 
        description="Identifiers tags list (e.g. ['v1', 'production', 'qlora'])"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full training arguments logged during the run creation"
    )
    registered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Registry catalog creation timestamp"
    )
