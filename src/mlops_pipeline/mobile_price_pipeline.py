import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_STAGES = [
    ("Data Loading", "01_load_data.py"),
    ("Raw Data Validation", "02_validate_data.py"),
    ("Preprocessing", "03_preprocessing.py"),
    ("Processed Output Validation", "04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "05_train_registry.py"),
    ("Model Evaluation", "06_model_evaluation.py"),
    ("Automated Model Lifecycle", "08_automate_lifecycle.py"),
    ("Production Inference", "09_inference.py"),
]


def run_pipeline(include_quality_gate=False):
    stages = list(BASE_STAGES)
    if include_quality_gate:
        stages.insert(6, ("Model Quality Gate", "07_quality_gate.py"))

    lab_label = "7-8" if include_quality_gate else "7"
    print("=" * 68)
    print(f"[INFO] Starting Lab {lab_label}: Mobile Price Classification MLOps Pipeline")
    print("=" * 68)
    run_log = {"project": "Mobile Price Classification", "labs": [7, 8] if include_quality_gate else [7], "stages": []}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for stage_name, script_name in stages:
        script_path = ROOT / "src" / "mlops_pipeline" / script_name
        print(f"\n[INFO] ---> Executing {stage_name}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT), env=env)
        status = "PASSED" if result.returncode == 0 else "FAILED"
        run_log["stages"].append({"stage": stage_name, "script": script_name, "status": status, "exit_code": result.returncode})
        print(f"[{status}] {stage_name}")
        if result.returncode != 0:
            run_log["overall_status"] = "FAILED"
            break
    else:
        run_log["overall_status"] = "PASSED"

    log_name = "mobile_price_lab8_run_log.json" if include_quality_gate else "mobile_price_lab7_run_log.json"
    log_path = ROOT / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print("\n" + "=" * 68)
    print(f"[INFO] Run log: {log_path}")
    print(f"[{'SUCCESS' if run_log['overall_status'] == 'PASSED' else 'ERROR'}] Lab {lab_label} pipeline {run_log['overall_status'].lower()}.")
    print("=" * 68)
    return run_log["overall_status"] == "PASSED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-quality-gate", action="store_true", help="Insert the Lab 8 quality gate before model lifecycle management")
    args = parser.parse_args()
    sys.exit(0 if run_pipeline(include_quality_gate=args.with_quality_gate) else 1)
