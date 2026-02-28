"""Logging and I/O utility helpers."""
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standard format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dirs(*dirs) -> None:
    """Create directories (and parents) if they do not already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
