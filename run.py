#!/usr/bin/env python3
"""
Run script for the Legal Sentence Boundary Detection Demo application.
"""

import os
import sys
import logging
import uvicorn
from app.logger import configure_uvicorn_logging, logger


def main():
    """Run the FastAPI application with uvicorn."""
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    # Map string log levels to uvicorn log levels
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    # Get log level, default to info
    numeric_level = log_level_map.get(log_level, logging.INFO)
    
    # Use port 8080 by default
    port = int(os.environ.get("PORT", 8080))
    
    # Log startup information
    logger.info(f"Starting Legal Sentence Boundary Detection Application")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Port: {port}")
    logger.info(f"Log file: {os.path.abspath('logs/app.log')}")
    
    # Configure uvicorn logging to use our log file
    configure_uvicorn_logging()
    
    # Configure uvicorn
    config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": port,
        "log_level": log_level,
        "reload": True,  # Enable auto-reload for development
        "reload_dirs": ["app", "templates", "static"],  # Watch these directories for changes
    }
    
    print(f"Starting server at http://localhost:{port}")
    print(f"Log level: {log_level}")
    print(f"Logs are being written to: {os.path.abspath('logs/app.log')}")
    print("Press Ctrl+C to stop")
    
    try:
        # Run the server
        uvicorn.run(**config)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        logger.exception("Server startup exception")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        print("\nShutting down server")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.exception("Unhandled exception")
        sys.exit(1)