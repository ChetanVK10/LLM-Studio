"""LLMOps Studio Logging Utility.

Configures a centralized logging mechanism. Supports separate rotating log files for
general application logic, model training, evaluation runs, and benchmarking suites.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

from backend.core.config import get_settings

# Global formatter format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def initialize_logging() -> None:
    """Initializes the multi-logger configuration.

    Sets up the RotatingFileHandlers for:
      - application.log
      - training.log
      - evaluation.log
      - benchmark.log
    along with StreamHandlers for standard console output.
    """
    settings = get_settings()
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Map specific logger namespaces to corresponding output files
    log_configs: Dict[str, str] = {
        "app": "application.log",
        "training": "training.log",
        "evaluation": "evaluation.log",
        "benchmark": "benchmark.log",
    }

    for logger_name, log_filename in log_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        log_filepath = (log_dir / log_filename).resolve()
        
        # 1. Clean up handlers targeting different filenames (e.g. redirected test paths)
        # and close handles to prevent resource leaks
        active_handlers: List[logging.Handler] = []
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler_path = Path(handler.baseFilename).resolve()
                if handler_path != log_filepath:
                    handler.close()
                    continue
            active_handlers.append(handler)
        logger.handlers = active_handlers

        # 2. Add StreamHandler if not already configured
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in logger.handlers
        )
        if not has_console:
            logger.addHandler(console_handler)

        # 3. Add RotatingFileHandler if not already configured
        has_file_handler = any(
            isinstance(h, RotatingFileHandler) and Path(h.baseFilename).resolve() == log_filepath
            for h in logger.handlers
        )
        if not has_file_handler:
            # Rotating file output: 5MB maximum file size with 5 backup generations
            file_handler = RotatingFileHandler(
                log_filepath,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        # Disable event propagation to the root logger to avoid double logging
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieves a pre-configured logger by namespace.

    If name is not "app", "training", "evaluation", or "benchmark", it returns a sub-logger
    propagating back to one of the primary targets.
    """
    return logging.getLogger(name)
