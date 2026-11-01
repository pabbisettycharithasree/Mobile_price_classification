import sys
import json
import numpy as np
from common import load_config, project_path, save_json

def main():
    print("[INFO] Lab 7 - Validating preprocessing outputs")
    config = load_config()
    pdir = project_path(config["data"]["processed_dir"])
    required = ["X_train_final.npy", "X_valid_final.npy", "y_train.npy", "y_valid.npy", "X_test_final.npy"]
    missing = [name for name in required if not (pdir / name).exists()]
    errors = [f"Missing artifact: {name}" for name in missing]

    if not errors:
        X_train = np.load(pdir / "X_train_final.npy")
        X_valid = np.load(pdir / "X_valid_final.npy")
        y_train = np.load(pdir / "y_train.npy")
        y_valid = np.load(pdir / "y_valid.npy")
        X_test = np.load(pdir / "X_test_final.npy")

        for name, matrix in [("X_train", X_train), ("X_valid", X_valid), ("X_test", X_test)]:
            if matrix.ndim != 2:
                errors.append(f"{name} must be a 2D matrix.")
            if not np.isfinite(matrix).all():
                errors.append(f"{name} contains NaN or infinite values.")

        if X_train.shape[1] != X_valid.shape[1] or X_train.shape[1] != X_test.shape[1]:
            errors.append("Feature dimension mismatch between train, validation and test matrices.")
        if len(X_train) != len(y_train):
            errors.append("X_train/y_train row count mismatch.")
        if len(X_valid) != len(y_valid):
            errors.append("X_valid/y_valid row count mismatch.")

        details = {
            "X_train_shape": list(X_train.shape),
            "X_valid_shape": list(X_valid.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_valid_shape": list(y_valid.shape),
            "missing_values": 0
        }
    else:
        details = {}

    passed = not errors
    save_json("artifacts/preprocessing_summary_report.json", {
        "stage": "processed_output_validation",
        "status": "PASSED" if passed else "FAILED",
        "details": details,
        "errors": errors
    })
    if passed:
        print("[SUCCESS] All processed matrices are valid.")
    else:
        for error in errors:
            print(f"[ERROR] {error}")
    return passed

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
