"""Lab 6: train the current champion candidate and register it in MLflow."""
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score

from src.mlflow_utils import setup_mlflow, REGISTERED_MODEL_NAME
from src.train import run as train_run
from src.utils import load_config, resolve


def run(config_path=None):
    setup_mlflow()
    cfg = load_config(config_path)
    best_name, _ = train_run(config_path)
    model_path = resolve(cfg["paths"]["models_dir"]) / "best_model.joblib"
    info = json.loads((resolve(cfg["paths"]["models_dir"]) / "best_model_info.json").read_text())
    model = joblib.load(model_path)
    test = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "test.csv")
    X, y = test.drop(columns=[cfg["target"]]), test[cfg["target"]]
    test_accuracy = float(accuracy_score(y, model.predict(X)))

    with mlflow.start_run(run_name=f"lab6_registry_{best_name}") as run:
        mlflow.log_param("model_name", best_name)
        mlflow.log_metric("cv_accuracy", float(info["cv_score"]))
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.set_tag("lab", "6")
        mlflow.set_tag("model_name", best_name)
        result = mlflow.sklearn.log_model(
            model, "model", registered_model_name=REGISTERED_MODEL_NAME
        )
        version = getattr(result, "registered_model_version", None)
        run_id = run.info.run_id

    # Resolve the version if the flavor did not return it directly.
    if version is None:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        versions = [v for v in versions if v.run_id == run_id]
        if not versions:
            raise RuntimeError("Model was logged but no registry version was found")
        version = max(versions, key=lambda v: int(v.version)).version

    # Store score metadata on the registered version for lifecycle comparison.
    client = mlflow.MlflowClient()
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "cv_accuracy", str(info["cv_score"]))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "test_accuracy", str(test_accuracy))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "source_run_id", run_id)
    print(f"[SUCCESS] Registered {REGISTERED_MODEL_NAME} version {version}")
    return int(version)


if __name__ == "__main__":
    run()
