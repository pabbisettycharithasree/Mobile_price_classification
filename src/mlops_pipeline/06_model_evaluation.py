import sys
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from common import load_config, project_path, save_json

def main():
    print("[INFO] Lab 7 - Evaluating the registered candidate model")
    config = load_config()
    pdir = project_path(config["data"]["processed_dir"])
    model_path = project_path(config["paths"]["model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    X_valid = np.load(pdir / "X_valid_final.npy")
    y_valid = np.load(pdir / "y_valid.npy")
    model = joblib.load(model_path)
    predictions = model.predict(X_valid)

    labels = config["data"]["class_labels"]
    metrics = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision_macro": float(precision_score(y_valid, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_valid, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_valid, predictions, average="macro", zero_division=0))
    }
    cm = confusion_matrix(y_valid, predictions, labels=labels)
    report = {
        "project": "Mobile Price Classification",
        "stage": "Lab 7 model evaluation",
        "evaluated_on": "stratified validation split from train.csv",
        "validation_rows": int(len(y_valid)),
        "class_labels": labels,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_valid, predictions, labels=labels, output_dict=True, zero_division=0
        )
    }
    save_json(config["paths"]["evaluation_report"], report)
    print("\n[RESULTS]")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("[SUCCESS] Evaluation report saved.")
    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        sys.exit(1)
