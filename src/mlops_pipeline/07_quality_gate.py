import sys
from common import load_config, project_path, save_json
import json


def run_quality_gate():
    print("[INFO] Lab 8 - Starting Mobile Price Model Quality Gate")
    config = load_config()
    report_path = project_path(config["paths"]["evaluation_report"])
    gate_path = config["paths"]["quality_gate_report"]
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    if not report_path.exists():
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    with report_path.open("r", encoding="utf-8") as f:
        evaluation = json.load(f)

    if metric_name not in evaluation.get("metrics", {}):
        print(f"[ERROR] Metric '{metric_name}' is missing from evaluation report.")
        return False

    actual_value = float(evaluation["metrics"][metric_name])
    passed = actual_value >= minimum_value
    gate_report = {
        "project": "Mobile Price Classification",
        "stage": "Lab 8 - Model Quality Gate",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
        "promotion_allowed": passed
    }
    save_json(gate_path, gate_report)
    print(f"[INFO] Quality Gate: {metric_name}={actual_value:.4f} | required>={minimum_value:.4f}")
    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
    else:
        print("[ERROR] Model Quality Gate FAILED. Pipeline stops before lifecycle promotion.")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_quality_gate() else 1)
