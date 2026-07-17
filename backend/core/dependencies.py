"""LLMOps Studio Dependency Injection Module.

Exposes mockable factory providers for all core services, registry catalogs, and settings,
allowing clean integration with frontend components or future FastAPI router contexts.
"""

from backend.core.config import Settings, get_settings as load_settings
from backend.datasets.loader import DatasetLoader
from backend.datasets.preview import DatasetPreviewGenerator
from backend.datasets.processor import DatasetProcessor
from backend.datasets.repository import SQLiteDatasetRepository
from backend.datasets.splitter import DatasetSplitter
from backend.datasets.token_estimator import HeuristicTokenEstimator
from backend.datasets.validator import DatasetValidator
from backend.experiment_tracking.tracker import ExperimentTracker
from backend.integrations.huggingface import HuggingFaceDatasetIngestor
from backend.services.benchmark_service import BenchmarkService
from backend.services.dataset_service import DatasetService
from backend.services.evaluation_service import EvaluationService
from backend.services.registry_service import RegistryService
from backend.services.training_service import TrainingService
from backend.storage.local_storage import LocalStorage

from backend.registry.catalog import SQLiteModelRegistry
from backend.training.runner import TrainingRunner

# Foundational singletons (lazy-initialized where dependencies are required)
_storage_manager = None
_dataset_repository = None
_dataset_service = None
_model_registry = None
_training_service = None

_evaluation_service = EvaluationService()
_benchmark_service = BenchmarkService()
_registry_service = RegistryService()
_experiment_tracker = ExperimentTracker()


def get_settings() -> Settings:
    """Provides the active configuration settings singleton."""
    return load_settings()


def get_storage_manager() -> LocalStorage:
    """Provides the LocalStorage manager singleton."""
    global _storage_manager
    if _storage_manager is None:
        settings = get_settings()
        _storage_manager = LocalStorage(settings.workspace_root)
    return _storage_manager


def get_dataset_repository() -> SQLiteDatasetRepository:
    """Provides the SQLite registry database repository singleton."""
    global _dataset_repository
    if _dataset_repository is None:
        settings = get_settings()
        db_path = settings.workspace_root / "data" / "datasets.db"
        _dataset_repository = SQLiteDatasetRepository(db_path)
    return _dataset_repository


def get_dataset_service() -> DatasetService:
    """Returns an instance of the DatasetService with all its submodules resolved."""
    global _dataset_service
    if _dataset_service is None:
        storage = get_storage_manager()
        repo = get_dataset_repository()
        
        loader = DatasetLoader(storage)
        validator = DatasetValidator()
        estimator = HeuristicTokenEstimator()
        preview = DatasetPreviewGenerator(estimator)
        splitter = DatasetSplitter()
        
        processor = DatasetProcessor(
            storage_manager=storage,
            repository=repo,
            loader=loader,
            validator=validator,
            preview_generator=preview,
            splitter=splitter
        )
        
        hf_ingestor = HuggingFaceDatasetIngestor()
        
        _dataset_service = DatasetService(
            processor=processor,
            repository=repo,
            hf_ingestor=hf_ingestor
        )
    return _dataset_service


def get_model_registry() -> SQLiteModelRegistry:
    """Provides the SQLiteModelRegistry catalog singleton."""
    global _model_registry
    if _model_registry is None:
        settings = get_settings()
        db_path = settings.workspace_root / "data" / "models.db"
        _model_registry = SQLiteModelRegistry(db_path)
    return _model_registry


def get_training_service() -> TrainingService:
    """Returns an instance of the TrainingService."""
    global _training_service
    if _training_service is None:
        storage = get_storage_manager()
        dataset_service = get_dataset_service()
        registry = get_model_registry()
        
        runner = TrainingRunner(
            storage_manager=storage,
            dataset_service=dataset_service,
            model_registry=registry
        )
        
        _training_service = TrainingService(runner=runner, registry=registry)
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
