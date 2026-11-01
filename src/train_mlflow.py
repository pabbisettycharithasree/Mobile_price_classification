"""Lab 4: train the baseline models and track the best model in MLflow."""
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score

from src.mlflow_utils import setup_mlflow, EXPERIMENT_NAME
from src.utils import load_config, resolve
from src.train import run as train_run


def run(config_path=None):
    setup_mlflow()
    cfg = load_config(config_path)

    # Reuse the tested Lab 3 training implementation so the tracked model
    # is exactly the same model used by the baseline pipeline.
    best_name, comparison = train_run(config_path)
    model_path = resolve(cfg["paths"]["models_dir"]) / "best_model.joblib"
    info_path = resolve(cfg["paths"]["models_dir"]) / "best_model_info.json"
    model = joblib.load(model_path)
    info = json.loads(info_path.read_text())

    test = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "test.csv")
    X_test = test.drop(columns=[cfg["target"]])
    y_test = test[cfg["target"]]
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run(run_name=f"lab4_{best_name}") as run:
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("random_state", cfg["training"]["random_state"])
        mlflow.log_param("cv_folds", cfg["training"]["cv_folds"])
        mlflow.log_metric("cv_accuracy", float(info["cv_score"]))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_artifact(str(model_path), artifact_path="model_files")
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("lab", "4")
        mlflow.set_tag("model_name", best_name)
        mlflow.set_tag("reproducible", "true")
        run_id = run.info.run_id

    print(f"[SUCCESS] MLflow run created: {run_id}")
    print(f"[INFO] Best model: {best_name}")
    print(f"[INFO] CV accuracy: {info['cv_score']:.4f}")
    print(f"[INFO] Test accuracy: {test_accuracy:.4f}")
    return run_id


if __name__ == "__main__":
    run()
