from pathlib import Path
import json
import os
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config_lab7.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories(config):
    for key in ("processed_dir",):
        project_path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    for key in ("model", "preprocessor"):
        project_path(config["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    for key in ("evaluation_report", "lifecycle_report"):
        project_path(config["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    project_path(config["paths"]["inference_output"]).parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts").mkdir(exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)


def save_json(path, payload):
    path = project_path(path) if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def expected_feature_columns(config):
    return list(config["data"]["feature_columns"])


def load_train_and_test(config):
    train_path = project_path(config["data"]["train_path"])
    test_path = project_path(config["data"]["test_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_train_test_schema(train_df, test_df, config):
    features = expected_feature_columns(config)
    target = config["data"]["target_column"]
    errors = []

    if train_df.empty:
        errors.append("Training dataset is empty.")
    if test_df.empty:
        errors.append("Test dataset is empty.")

    expected_train = set(features + [target])
    expected_test = set(features)
    if set(train_df.columns) != expected_train:
        missing = sorted(expected_train - set(train_df.columns))
        extra = sorted(set(train_df.columns) - expected_train)
        errors.append(f"Training columns mismatch. Missing={missing}; extra={extra}")

    test_features = set(test_df.columns) - {config["data"].get("id_column", "id")}
    if test_features != expected_test:
        missing = sorted(expected_test - test_features)
        extra = sorted(test_features - expected_test)
        errors.append(f"Test feature columns mismatch. Missing={missing}; extra={extra}")

    if train_df.columns.duplicated().any():
        errors.append("Training dataset has duplicate column names.")
    if test_df.columns.duplicated().any():
        errors.append("Test dataset has duplicate column names.")

    if target in train_df.columns:
        if train_df[target].isna().any():
            errors.append(f"Target column '{target}' contains missing values.")
        allowed = set(config["data"]["class_labels"])
        actual = set(pd.to_numeric(train_df[target], errors="coerce").dropna().astype(int).unique())
        if actual != allowed:
            errors.append(f"Target classes must be {sorted(allowed)}, found {sorted(actual)}.")

    for col in features:
        if col in train_df.columns:
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                errors.append(f"Training feature '{col}' must be numeric.")
            elif train_df[col].isna().any():
                errors.append(f"Training feature '{col}' contains missing values.")
        if col in test_df.columns:
            if not pd.api.types.is_numeric_dtype(test_df[col]):
                errors.append(f"Test feature '{col}' must be numeric.")
            elif test_df[col].isna().any():
                errors.append(f"Test feature '{col}' contains missing values.")

    if errors:
        return False, errors
    return True, []


def configure_mlflow(config):
    import mlflow
    uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlflow", {}).get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return mlflow
