import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common import load_config, project_path, configure_mlflow

def main():
    print("[INFO] Lab 7 - Training model and registering with MLflow")
    config = load_config()
    mlflow_client = configure_mlflow(config)
    pdir = project_path(config["data"]["processed_dir"])
    X_train = np.load(pdir / "X_train_final.npy")
    y_train = np.load(pdir / "y_train.npy")
    X_valid = np.load(pdir / "X_valid_final.npy")
    y_valid = np.load(pdir / "y_valid.npy")

    model_cfg = config["model"]
    params = {
        "n_estimators": int(model_cfg["n_estimators"]),
        "max_depth": model_cfg.get("max_depth"),
        "min_samples_split": int(model_cfg["min_samples_split"]),
        "random_state": int(model_cfg["random_state"]),
        "class_weight": model_cfg.get("class_weight"),
        "n_jobs": -1
    }

    with mlflow.start_run(run_name="MobilePrice_RandomForest") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestClassifier")
        mlflow.log_param("problem_type", "multiclass_classification")
        mlflow.log_param("target_column", config["data"]["target_column"])
        mlflow.log_param("validation_strategy", "stratified_holdout")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        metrics = {
            "accuracy": float(accuracy_score(y_valid, pred)),
            "precision_macro": float(precision_score(y_valid, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_valid, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_valid, pred, average="macro", zero_division=0))
        }
        mlflow.log_metrics(metrics)

        model_path = project_path(config["paths"]["model"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        preprocessor_path = project_path(config["paths"]["preprocessor"])
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mobile_price_random_forest",
            registered_model_name=config["registry"]["model_name"]
        )

        print(f"[SUCCESS] Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"[SUCCESS] Macro F1: {metrics['f1_macro']:.4f}")
        print(f"[SUCCESS] Registered model: {config['registry']['model_name']}")
        print(f"[INFO] MLflow run ID: {run.info.run_id}")

    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:
        print(f"[ERROR] Model training/registration failed: {exc}")
        sys.exit(1)
