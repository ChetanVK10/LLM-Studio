"""LLMOps Studio Custom Exceptions.

Defines the core hierarchy of domain-specific exceptions to distinguish system errors
from validation and execution failures.
"""


class LLMOpsStudioError(Exception):
    """Base exception class for all LLMOps Studio errors."""
    pass


class ConfigurationError(LLMOpsStudioError):
    """Raised when environment variables or settings files fail to load or validate."""
    pass


class DatasetError(LLMOpsStudioError):
    """Base exception for all dataset-related issues."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a requested dataset path or register key does not exist."""
    pass


class DatasetValidationError(DatasetError):
    """Raised when an uploaded or processed dataset fails schema validation."""
    pass


class InvalidSplitRatioError(DatasetError):
    """Raised when train/validation/test split ratios do not sum to 1.0 or are out of bounds."""
    pass


class DatasetLoaderError(DatasetError):
    """Raised when parsing or reading dataset files encounters format or structural issues."""
    pass


class StorageError(LLMOpsStudioError):
    """Raised when reading, writing, or deleting files in storage layer fails."""
    pass


class TrainingError(LLMOpsStudioError):
    """Base exception for training workflow failures."""
    pass


class TrainingFailedError(TrainingError):
    """Raised when the fine-tuning training loop terminates due to runtime exceptions."""
    pass


class HyperparameterError(TrainingError):
    """Raised when provided training arguments do not fit parameters boundary."""
    pass


class EvaluationError(LLMOpsStudioError):
    """Base exception for evaluation workflow failures."""
    pass


class EvaluationFailedError(EvaluationError):
    """Raised when a model evaluation run fails to compute."""
    pass


class MetricNotSupportedError(EvaluationError):
    """Raised when an requested evaluation metric is unrecognized by the runner."""
    pass


class BenchmarkError(LLMOpsStudioError):
    """Base exception for benchmarking suite errors."""
    pass


class BenchmarkFailedError(BenchmarkError):
    """Raised when a benchmark metrics collection fails to terminate correctly."""
    pass


class HardwareAccessError(BenchmarkError):
    """Raised when requested device (e.g. CUDA GPU) is inaccessible or runs out of memory."""
    pass


class RegistryError(LLMOpsStudioError):
    """Base exception for registry cataloging issues."""
    pass


class CheckpointNotFoundError(RegistryError):
    """Raised when looking up adapter weights or tags that are not registered."""
    pass


class ModelNameConflictError(RegistryError):
    """Raised when attempting to overwrite an active registry slot with a duplicate name."""
    pass


class ExperimentTrackingError(LLMOpsStudioError):
    """Raised when metrics loggers or experiment run lifecycles hit inconsistent states."""
    pass


class EvaluationError(LLMOpsStudioError):
    """Raised when evaluation pipelines encounter validation or execution failures."""
    pass

