import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(timestamp, log_dir="logs", level=logging.INFO):
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Dynamic log file per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers (important when scripts are rerun in some environments)
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Terminal handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Log file: {logfile}")

    return logger