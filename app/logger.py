"""
Centralized logging configuration for the application.
"""

import logging
import logging.handlers
import os
import sys
import pathlib
from typing import Optional


def setup_logger(
    name: str = "legal-sentence-demo",
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.
    
    Args:
        name: The name of the logger
        log_file: Path to the log file. If None, uses default path
        level: The logging level for the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default log file path
    if log_file is None:
        base_dir = pathlib.Path(__file__).parent.parent
        log_dir = base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "app.log")
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create the default application logger
logger = setup_logger()

# Configure uvicorn access logger to use our file handler
def configure_uvicorn_logging() -> None:
    """Configure uvicorn loggers to use our log file."""
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_error = logging.getLogger("uvicorn.error")
    
    # Get our file handler
    app_logger = logging.getLogger("legal-sentence-demo")
    for handler in app_logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            # Create new handler with same configuration
            file_handler = logging.handlers.RotatingFileHandler(
                handler.baseFilename,
                maxBytes=handler.maxBytes,
                backupCount=handler.backupCount,
                encoding=handler.encoding
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - UVICORN - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            
            # Add to uvicorn loggers
            uvicorn_access.addHandler(file_handler)
            uvicorn_error.addHandler(file_handler)
            break