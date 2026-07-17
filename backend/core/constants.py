"""LLMOps Studio Core Constants.

Centralizes Enums and frozen configuration metadata values to avoid hardcoded string
literals across components.
"""

from enum import Enum


class AppEnv(str, Enum):
    """Supported application runtime environment stages."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class ModelProvider(str, Enum):
    """Supported large language model API and engine providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ExperimentStatus(str, Enum):
    """Lifecycle statuses of an ML tracking run or workflow execution."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingState(str, Enum):
    """Granular states of the LLM training pipeline."""
    INITIALIZING = "initializing"
    TOKENIZING = "tokenizing"
    TRAINING = "training"
    SAVING_CHECKPOINT = "saving_checkpoint"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    """Lifecycle statuses of a batch evaluation workload."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class ArtifactType(str, Enum):
    """Categorized outputs and generated assets saved to filesystem registry."""
    CHECKPOINT = "checkpoint"
    EVALUATION_REPORT = "evaluation_report"
    BENCHMARK_SUMMARY = "benchmark_summary"
    EXPORTED_MODEL = "exported_model"


class FileExtension(str, Enum):
    """Standard file extensions used across datasets and checkpoints."""
    JSON = ".json"
    YAML = ".yaml"
    YML = ".yml"
    PARQUET = ".parquet"
    SAFETENSORS = ".safetensors"
    LOG = ".log"
    MD = ".md"


class DirectoryName(str, Enum):
    """Standard subdirectory structures inside data and artifacts partitions."""
    # Data partitions
    DATA_RAW = "raw"
    DATA_PROCESSED = "processed"
    DATA_CACHE = "cache"
    DATA_UPLOADS = "uploads"

    # Artifacts partitions
    ARTIFACTS_CHECKPOINTS = "checkpoints"
    ARTIFACTS_EVALUATIONS = "evaluations"
    ARTIFACTS_REPORTS = "reports"
    ARTIFACTS_BENCHMARKS = "benchmarks"
    ARTIFACTS_EXPORTS = "exports"


class DatasetLifecycleState(str, Enum):
    """Representing the active step in a dataset's import lifecycle."""
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    PROFILED = "profiled"
    SPLIT = "split"
    REGISTERED = "registered"
    READY = "ready"


class ValidationSeverity(str, Enum):
    """Categorized levels of dataset validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
