"""Lab 6: generate a JSON report of registered model versions and Champion."""
import json
from datetime import datetime, timezone
import mlflow
from src.mlflow_utils import setup_mlflow, REGISTERED_MODEL_NAME
from src.utils import resolve, load_config


def run(config_path=None):
    setup_mlflow()
    client = mlflow.MlflowClient()
    versions = list(client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
    if not versions:
        raise RuntimeError("No registered model versions found")
    champion = None
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Champion")
    except Exception:
        pass
    report = {
        "project": load_config(config_path)["project_name"],
        "registered_model": REGISTERED_MODEL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_version": str(champion.version) if champion else None,
        "versions": [
            {"version": str(v.version), "stage": v.current_stage,
             "run_id": v.run_id, "tags": dict(v.tags)}
            for v in sorted(versions, key=lambda x: int(x.version))
        ],
    }
    out = resolve("reports") / "registry_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[SUCCESS] Registry report written to {out}")
    return report


if __name__ == "__main__":
    run()
