import sys
import json
import mlflow
from mlflow.tracking import MlflowClient
from common import load_config, configure_mlflow, save_json

def main():
    print("[INFO] Lab 7 - Managing model registry lifecycle")
    config = load_config()
    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_name = config["registry"]["decision_metric"]

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for '{model_name}'.")

    candidates = [v for v in versions if v.current_stage == "None"]
    if not candidates:
        candidates = [v for v in versions if v.current_stage == "Staging"]

    for version in candidates:
        if version.current_stage == "None":
            client.transition_model_version_stage(
                name=model_name, version=version.version, stage="Staging",
                archive_existing_versions=False
            )
            print(f"[INFO] Model version {version.version} moved to Staging.")

    versions = client.search_model_versions(f"name='{model_name}'")
    staging = [v for v in versions if v.current_stage == "Staging"]
    production = [v for v in versions if v.current_stage == "Production"]

    def metric_for(version):
        run = client.get_run(version.run_id)
        return float(run.data.metrics.get(metric_name, -1.0))

    current_prod = max(production, key=lambda v: int(v.version)) if production else None
    best_candidate = max(staging, key=metric_for) if staging else None

    promoted = False
    decision = "NO_STAGING_CANDIDATE"
    if best_candidate is not None:
        candidate_score = metric_for(best_candidate)
        production_score = metric_for(current_prod) if current_prod else None
        if current_prod is None or candidate_score > production_score:
            client.transition_model_version_stage(
                name=model_name,
                version=best_candidate.version,
                stage="Production",
                archive_existing_versions=True
            )
            promoted = True
            decision = "PROMOTED_TO_PRODUCTION"
        else:
            decision = "KEPT_EXISTING_PRODUCTION"

    versions_after = client.search_model_versions(f"name='{model_name}'")
    production_after = [v for v in versions_after if v.current_stage == "Production"]
    report = {
        "model_name": model_name,
        "decision_metric": metric_name,
        "decision": decision,
        "promoted": promoted,
        "candidate_version": int(best_candidate.version) if best_candidate else None,
        "candidate_metric": metric_for(best_candidate) if best_candidate else None,
        "production_version_after": int(production_after[0].version) if production_after else None
    }
    save_json(config["paths"]["lifecycle_report"], report)
    print(f"[SUCCESS] Lifecycle decision: {decision}")
    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:
        print(f"[ERROR] Model lifecycle failed: {exc}")
        sys.exit(1)
