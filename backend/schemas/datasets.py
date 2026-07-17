"""Datasets Schemas.

Defines Pydantic structures for tracking dataset ingestion profiles.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata schema representing structural details of registered dataset."""
    dataset_name: str = Field(
        ..., 
        description="Unique registry search key for the dataset"
    )
    raw_path: str = Field(
        ..., 
        description="Filer system storage target for raw unmodified files"
    )
    processed_path: Optional[str] = Field(
        None, 
        description="Folder path location storing cleaned and tokenized outputs"
    )
    num_samples: int = Field(
        0, 
        description="Total sample rows count found in dataset files"
    )
    features: List[str] = Field(
        default_factory=list, 
        description="Columns schema identifiers parsed from file header mappings"
    )
    split_ratio: str = Field(
        "train/val/test", 
        description="Text representation of dataset segment percentages"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp indicating initial dataset ingestion"
    )
