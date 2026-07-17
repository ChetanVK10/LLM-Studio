"""Unit tests for multi-file rotating logger.

Validates that logs written to specific namespaces route correctly to their respective files.
"""

from backend.core.config import get_settings
from backend.utils.logging import get_logger, initialize_logging


def test_logger_routing(tmp_path, monkeypatch) -> None:
    """Verifies that app and training logs write strictly to distinct targets."""
    # Construct a modified Settings copy with a redirected log directory
    custom_settings = get_settings().model_copy(update={"log_dir": tmp_path})
    
    # Mock the config settings getter inside logging utility
    monkeypatch.setattr("backend.utils.logging.get_settings", lambda: custom_settings)
    
    try:
        # Re-initialize logging layout under temporary directory
        initialize_logging()
        
        app_logger = get_logger("app")
        training_logger = get_logger("training")
        
        msg_app = "VERIFY_ROUTING_APP_LOG"
        msg_train = "VERIFY_ROUTING_TRAIN_LOG"
        
        app_logger.info(msg_app)
        training_logger.info(msg_train)
        
        app_file = tmp_path / "application.log"
        train_file = tmp_path / "training.log"
        
        assert app_file.exists()
        assert train_file.exists()
        
        with open(app_file, "r", encoding="utf-8") as f:
            app_content = f.read()
            assert msg_app in app_content
            assert msg_train not in app_content
            
        with open(train_file, "r", encoding="utf-8") as f:
            train_content = f.read()
            assert msg_train in train_content
            assert msg_app not in train_content
            
    finally:
        # Re-initialize default logging paths
        initialize_logging()
