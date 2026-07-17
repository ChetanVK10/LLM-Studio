"""Experiments Schemas.

Defines Pydantic structures for experiment metadata tracking.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExperimentMetadata(BaseModel):
    """Metadata schema representing parameters, statuses, and logs of an experiment run."""
    experiment_id: str = Field(
        ..., 
        description="Unique system experiment group identifier"
    )
    run_id: str = Field(
        ..., 
        description="Unique logging run session identifier"
    )
    name: str = Field(
        ..., 
        description="Friendly search name for the run session"
    )
    status: str = Field(
        ..., 
        description="Lifecycle tracking state of the run"
    )
    parameters: Dict[str, Any] = Field(
        ..., 
        description="Input hyperparameter and setup logs values mapping"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Evaluated tracking metrics compiled during execution"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp indicating run registration time"
    )
    completed_at: Optional[datetime] = Field(
        None, 
        description="Timestamp indicating run completion time"
    )
