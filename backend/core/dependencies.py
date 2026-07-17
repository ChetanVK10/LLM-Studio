"""LLMOps Studio Dependency Injection Module.

Exposes mockable factory providers for all core services, registry catalogs, and settings,
allowing clean integration with frontend components or future FastAPI router contexts.
"""

from backend.core.config import Settings, get_settings as load_settings
from backend.services.dataset_service import DatasetService
from backend.services.training_service import TrainingService
from backend.services.evaluation_service import EvaluationService
from backend.services.benchmark_service import BenchmarkService
from backend.services.registry_service import RegistryService
from backend.experiment_tracking.tracker import ExperimentTracker

# Initialize foundational singletons
_dataset_service = DatasetService()
_training_service = TrainingService()
_evaluation_service = EvaluationService()
_benchmark_service = BenchmarkService()
_registry_service = RegistryService()
_experiment_tracker = ExperimentTracker()


def get_settings() -> Settings:
    """Provides the active configuration settings singleton."""
    return load_settings()


def get_dataset_service() -> DatasetService:
    """Returns an instance of the DatasetService."""
    return _dataset_service


def get_training_service() -> TrainingService:
    """Returns an instance of the TrainingService."""
    return _training_service


def get_evaluation_service() -> EvaluationService:
    """Returns an instance of the EvaluationService."""
    return _evaluation_service


def get_benchmark_service() -> BenchmarkService:
    """Returns an instance of the BenchmarkService."""
    return _benchmark_service


def get_registry_service() -> RegistryService:
    """Returns an instance of the RegistryService."""
    return _registry_service


def get_experiment_tracker() -> ExperimentTracker:
    """Returns an instance of the ExperimentTracker."""
    return _experiment_tracker
