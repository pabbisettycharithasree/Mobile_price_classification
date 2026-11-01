"""Shared helpers: config loading, path resolution, logging."""
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(relative_path) -> Path:
    """Resolve a config path relative to the project root and make sure its folder exists."""
    p = ROOT / relative_path
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    (ROOT / "logs").mkdir(exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / log_file)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
