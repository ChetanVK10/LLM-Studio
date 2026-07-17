"""Datasets Schemas.

Defines Pydantic structures for tracking dataset ingestion profiles,
validation issues, data summaries, and splitting options.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.core.constants import DatasetLifecycleState, ValidationSeverity


class ValidationIssue(BaseModel):
    """Details of a single validation finding."""

    severity: ValidationSeverity = Field(
        ..., 
        description="Severity classification: INFO, WARNING, or ERROR"
    )
    message: str = Field(
        ..., 
        description="Detailed description of the validation issue"
    )
    affected_field: Optional[str] = Field(
        None, 
        description="Name of the column or parameter affected, if applicable"
    )


class DatasetValidationReport(BaseModel):
    """Aggregated validation findings and criteria state."""

    is_valid: bool = Field(
        ..., 
        description="Indicates if the dataset is clean and conforms to required schema rules"
    )
    issues: List[ValidationIssue] = Field(
        default_factory=list, 
        description="List of validation warnings and error findings"
    )


class DatasetProfile(BaseModel):
    """Statistical summary profile representing row contents and estimated counts."""

    rows_count: int = Field(
        0, 
        description="Total sample rows found in raw files"
    )
    columns: List[str] = Field(
        default_factory=list, 
        description="List of columns parsed from schemas"
    )
    duplicate_count: int = Field(
        0, 
        description="Number of identical sample records detected"
    )
    missing_values: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of missing/null values grouped per column key"
    )
    avg_prompt_length: float = Field(
        0.0, 
        description="Average length of instructions/prompts"
    )
    avg_response_length: float = Field(
        0.0, 
        description="Average length of outputs responses text"
    )
    estimated_tokens: int = Field(
        0, 
        description="Total estimated token weight computed via estimators"
    )


class DatasetSplitConfig(BaseModel):
    """Configuration ratios used to slice data partitions."""

    train_ratio: float = Field(
        0.8, 
        description="Train set portion float value (0.0 to 1.0)"
    )
    val_ratio: float = Field(
        0.1, 
        description="Validation set portion float value (0.0 to 1.0)"
    )
    test_ratio: float = Field(
        0.1, 
        description="Test set portion float value (0.0 to 1.0)"
    )
    random_seed: int = Field(
        42, 
        description="Seed parameter to guarantee reproducible random partitionings"
    )


class DatasetMetadata(BaseModel):
    """Metadata schema representing structural and status details of a registered dataset."""

    dataset_id: str = Field(
        ..., 
        description="Content SHA-256 identifier key"
    )
    dataset_name: str = Field(
        ..., 
        description="Unique name registered for selection matching"
    )
    raw_path: str = Field(
        ..., 
        description="Workspace path targeting raw file upload"
    )
    processed_path: Optional[str] = Field(
        None, 
        description="Folder path location storing processed splits parquet files"
    )
    lifecycle_state: DatasetLifecycleState = Field(
        default=DatasetLifecycleState.UPLOADED,
        description="Active state in the dataset's ingestion cycle"
    )
    version: int = Field(
        1, 
        description="Version check parameter tag"
    )
    source: str = Field(
        "local", 
        description="Ingestion path source (e.g. 'local', 'huggingface')"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp indicating raw registry date"
    )
    profile: Optional[DatasetProfile] = None
    validation_report: Optional[DatasetValidationReport] = None
    split_config: Optional[DatasetSplitConfig] = None
