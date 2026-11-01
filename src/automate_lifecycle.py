"""Lab 6: compare registered versions and assign the Champion alias."""
import mlflow
from src.mlflow_utils import setup_mlflow, REGISTERED_MODEL_NAME


def run():
    setup_mlflow()
    client = mlflow.MlflowClient()
    try:
        versions = list(client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
    except Exception as exc:
        raise RuntimeError("Could not read the MLflow model registry") from exc
    if not versions:
        raise RuntimeError("No registered model versions found")
    def score(v):
        try: return float(v.tags.get("test_accuracy", v.tags.get("cv_accuracy", "-1")))
        except Exception: return -1.0
    champion = max(versions, key=score)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Champion", champion.version)
    print(f"[SUCCESS] Champion alias -> version {champion.version} (score={score(champion):.4f})")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"[INFO] Version {v.version}: score={score(v):.4f}, run={v.run_id}")
    return champion.version


if __name__ == "__main__":
    run()
