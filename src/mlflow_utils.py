"""Shared MLflow configuration for Labs 4 and 6."""
from pathlib import Path
import mlflow

from src.utils import ROOT

MLFLOW_DB = ROOT / "mlflow.db"
EXPERIMENT_NAME = "Mobile Price Classification"
REGISTERED_MODEL_NAME = "MobilePriceClassifier"


def setup_mlflow():
    """Use a project-local SQLite backend so no separate MLflow server is required."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
