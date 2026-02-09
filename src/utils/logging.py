"""Simple logging utilities for console, file, and JSON metrics."""

import json
import logging
from pathlib import Path


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """Create a logger that writes to console and optionally to a file.

    Args:
        name: Logger name.
        log_dir: If provided, also write logs to a file in this directory.

    Returns:
        Configured Python logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{name}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_metrics(filepath: str, metrics_dict: dict) -> None:
    """Append a metrics dictionary as a JSON line to a file.

    Args:
        filepath: Path to the JSON-lines file.
        metrics_dict: Dictionary of metric names to values.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics_dict) + "\n")
