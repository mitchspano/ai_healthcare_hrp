"""Logging infrastructure with configurable levels and outputs."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys
from datetime import datetime


class PipelineLogger:
    """Centralized logging system for the diabetes LSTM pipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the logging system.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config or {}
        self.loggers = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        # Get configuration values with defaults
        log_level = self.config.get("level", "INFO").upper()
        log_format = self.config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file = self.config.get("file", "logs/pipeline.log")
        max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        backup_count = self.config.get("backup_count", 5)

        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log the initialization
        logging.info(
            f"Logging system initialized - Level: {log_level}, File: {log_file}"
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.

        Args:
            name: Logger name (typically module name)

        Returns:
            Logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger

        return self.loggers[name]

    def set_level(self, level: str) -> None:
        """
        Change the logging level for all loggers.

        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = level.upper()
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))

        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(getattr(logging, log_level))

        logging.info(f"Logging level changed to {log_level}")

    def add_file_handler(self, filename: str, level: str = "INFO") -> None:
        """
        Add an additional file handler.

        Args:
            filename: Path to the log file
            level: Logging level for this handler
        """
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            self.config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logging.info(f"Added file handler: {filename} (level: {level})")

    def log_execution_start(self, component: str, **kwargs) -> None:
        """
        Log the start of a pipeline component execution.

        Args:
            component: Name of the component being executed
            **kwargs: Additional context information
        """
        logger = self.get_logger(component)
        context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(
            f"Starting execution of {component}"
            + (f" with {context}" if context else "")
        )

    def log_execution_end(
        self, component: str, duration: Optional[float] = None, **kwargs
    ) -> None:
        """
        Log the end of a pipeline component execution.

        Args:
            component: Name of the component that finished
            duration: Execution duration in seconds
            **kwargs: Additional context information
        """
        logger = self.get_logger(component)
        context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        duration_str = f" (duration: {duration:.2f}s)" if duration else ""
        logger.info(
            f"Completed execution of {component}{duration_str}"
            + (f" with {context}" if context else "")
        )

    def log_error(self, component: str, error: Exception, **kwargs) -> None:
        """
        Log an error with context information.

        Args:
            component: Name of the component where error occurred
            error: The exception that occurred
            **kwargs: Additional context information
        """
        logger = self.get_logger(component)
        context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.error(
            f"Error in {component}: {str(error)}"
            + (f" (context: {context})" if context else ""),
            exc_info=True,
        )

    def log_metrics(self, component: str, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics.

        Args:
            component: Name of the component
            metrics: Dictionary of metrics to log
        """
        logger = self.get_logger(component)
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"Metrics for {component}: {metrics_str}")

    def create_experiment_log(self, experiment_name: str) -> str:
        """
        Create a separate log file for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/experiments/{experiment_name}_{timestamp}.log"

        self.add_file_handler(log_filename, level="DEBUG")

        logger = self.get_logger("experiment")
        logger.info(f"Started experiment: {experiment_name}")

        return log_filename


# Global logger instance
_global_logger = None


def setup_logging(config: Dict[str, Any] = None) -> PipelineLogger:
    """
    Set up the global logging system.

    Args:
        config: Logging configuration dictionary

    Returns:
        Global PipelineLogger instance
    """
    global _global_logger
    _global_logger = PipelineLogger(config)
    return _global_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (if None, uses calling module name)

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = PipelineLogger()

    if name is None:
        # Try to get the calling module name
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    return _global_logger.get_logger(name)


def log_execution(component: str):
    """
    Decorator to automatically log function execution.

    Args:
        component: Name of the component being executed
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(component)
            start_time = datetime.now()

            try:
                logger.info(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {func.__name__} (duration: {duration:.2f}s)")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Error in {func.__name__} after {duration:.2f}s: {str(e)}",
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
