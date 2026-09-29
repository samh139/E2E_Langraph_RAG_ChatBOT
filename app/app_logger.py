import logging
from logging import Logger
from pathlib import Path


class LoggerFactory:

    @staticmethod
    def get_logger(logger_name: str) -> Logger:
        # Create/Get logger
        logger = logging.getLogger(logger_name)

        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "app.log"

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # File Handler
        file_handler = logging.FileHandler(
            log_file,
            mode="a",          # Append to existing log file
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger