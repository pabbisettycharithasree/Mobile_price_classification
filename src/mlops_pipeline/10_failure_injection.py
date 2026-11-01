"""Lab 9 failure injection tests for the Mobile Price Classification pipeline.

Run only after the normal Lab 7-8 pipeline has completed successfully.
All modified files are restored in finally blocks.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, load_config, project_path, configure_mlflow, save_json


def run_script(relative_path):
    script = PROJECT_ROOT / relative_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env)
    return result.returncode


def remove_backup(path):
    if path.exists():
        path.unlink()


def scenario_missing_training_data(config):
    print("\n[LAB 9] Scenario 1: Missing training dataset")
    path = project_path(config["data"]["train_path"])
    backup = Path(str(path) + ".lab9_backup")
    if not path.exists():
        return {"scenario": "missing_training_data", "expected": "HALT", "actual": "NOT_TESTED", "passed": False}
    remove_backup(backup)
    path.replace(backup)
    try:
        code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = code != 0
        return {"scenario": "missing_training_data", "expected": "HALT", "actual": "HALT" if passed else "CONTINUED", "exit_code": code, "passed": passed}
    finally:
        if backup.exists():
            if path.exists():
                path.unlink()
            backup.replace(path)


def scenario_invalid_datatype(config):
    print("\n[LAB 9] Scenario 2: Invalid feature datatype")
    path = project_path(config["data"]["train_path"])
    backup = Path(str(path) + ".lab9_backup")
    shutil.copy2(path, backup)
    try:
        df = pd.read_csv(path)
        feature = config["data"]["feature_columns"][0]
        df[feature] = df[feature].astype(object)
        df.loc[df.index[0], feature] = "INVALID_NUMBER"
        df.to_csv(path, index=False)
        code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = code != 0
        return {"scenario": "invalid_datatype", "feature": feature, "expected": "HALT", "actual": "HALT" if passed else "CONTINUED", "exit_code": code, "passed": passed}
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)


def scenario_schema_mismatch(config):
    print("\n[LAB 9] Scenario 3: Training schema mismatch")
    path = project_path(config["data"]["train_path"])
    backup = Path(str(path) + ".lab9_backup")
    shutil.copy2(path, backup)
    try:
        df = pd.read_csv(path)
        feature = config["data"]["feature_columns"][-1]
        df = df.drop(columns=[feature])
        df.to_csv(path, index=False)
        code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = code != 0
        return {"scenario": "schema_mismatch", "removed_column": feature, "expected": "HALT", "actual": "HALT" if passed else "CONTINUED", "exit_code": code, "passed": passed}
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)


def scenario_corrupted_processed_output(config):
    print("\n[LAB 9] Scenario 4: Corrupted processed output")
    path = project_path(config["data"]["processed_dir"]) / "X_train_final.npy"
    backup = Path(str(path) + ".lab9_backup")
    if not path.exists():
        return {"scenario": "corrupted_processed_output", "expected": "HALT", "actual": "NOT_TESTED", "passed": False}
    shutil.copy2(path, backup)
    try:
        values = np.load(path)
        values[0, 0] = np.nan
        np.save(path, values)
        code = run_script("src/mlops_pipeline/04_validate_outputs.py")
        passed = code != 0
        return {"scenario": "corrupted_processed_output", "injected_value": "NaN", "expected": "HALT", "actual": "HALT" if passed else "CONTINUED", "exit_code": code, "passed": passed}
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)


def scenario_missing_model_recovery(config):
    print("\n[LAB 9] Scenario 5: Missing local model and MLflow Production recovery")
    model_path = project_path(config["paths"]["model"])
    backup = Path(str(model_path) + ".lab9_backup")
    if not model_path.exists():
        return {"scenario": "missing_model_artifact", "expected": "RECOVER", "actual": "NOT_TESTED", "passed": False}

    remove_backup(backup)
    shutil.copy2(model_path, backup)
    model_path.unlink()
    failure_detected = False
    recovery_succeeded = False
    error = None
    try:
        failure_code = run_script("src/mlops_pipeline/06_model_evaluation.py")
        failure_detected = failure_code != 0

        configure_mlflow(config)
        model_name = config["registry"]["model_name"]
        production_uri = f"models:/{model_name}/Production"
        print(f"[INFO] Restoring registered model from {production_uri}")
        recovered_model = mlflow.sklearn.load_model(production_uri)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(recovered_model, model_path)

        recovery_code = run_script("src/mlops_pipeline/06_model_evaluation.py")
        recovery_succeeded = recovery_code == 0
        passed = failure_detected and recovery_succeeded
        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "RECOVERED" if passed else "RECOVERY_FAILED",
            "failure_detected": failure_detected,
            "recovery_succeeded": recovery_succeeded,
            "passed": passed
        }
    except Exception as exc:
        error = str(exc)
        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "RECOVERY_FAILED",
            "failure_detected": failure_detected,
            "recovery_succeeded": False,
            "error": error,
            "passed": False
        }
    finally:
        if backup.exists():
            if model_path.exists():
                model_path.unlink()
            backup.replace(model_path)


def run_failure_tests():
    print("[INFO] ==================================================")
    print("[INFO] Lab 9: Mobile Price Failure Handling and Recovery")
    print("[INFO] ==================================================")
    config = load_config()
    repetitions = int(config.get("lab9", {}).get("repetitions", 1))
    scenarios = [
        scenario_missing_training_data,
        scenario_invalid_datatype,
        scenario_schema_mismatch,
        scenario_corrupted_processed_output,
        scenario_missing_model_recovery,
    ]
    runs = []
    for repetition in range(1, repetitions + 1):
        print(f"\n[INFO] Lab 9 repetition {repetition}/{repetitions}")
        results = [scenario(config) for scenario in scenarios]
        runs.append({"repetition": repetition, "results": results, "passed": all(x["passed"] for x in results)})

    report = {
        "project": "Mobile Price Classification",
        "stage": "Lab 9 - Failure Handling and Recovery",
        "repetitions": repetitions,
        "runs": runs,
        "overall_status": "PASSED" if all(x["passed"] for x in runs) else "FAILED"
    }
    save_json(config["paths"]["failure_report"], report)
    print(f"\n[{report['overall_status']}] Lab 9 failure tests")
    print(f"[INFO] Report saved to {project_path(config['paths']['failure_report'])}")
    return report["overall_status"] == "PASSED"


if __name__ == "__main__":
    try:
        sys.exit(0 if run_failure_tests() else 1)
    except Exception as exc:
        print(f"[ERROR] Lab 9 failure tests crashed: {exc}")
        sys.exit(1)
