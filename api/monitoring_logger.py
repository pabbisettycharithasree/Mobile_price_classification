from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "inference" / "inference_log.jsonl"


def log_prediction(features, prediction, model_version="local", elapsed_ms=None):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": int(prediction),
        "model_version": model_version,
        "elapsed_ms": elapsed_ms,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")