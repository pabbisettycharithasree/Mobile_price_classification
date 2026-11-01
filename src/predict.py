"""Predict price ranges for new/unlabelled phones (e.g. Kaggle test.csv).

Usage:  python -m src.predict [path/to/file.csv]
Output: outputs/predictions.csv
"""
import sys

import joblib
import pandas as pd

from src.preprocess import engineer_features, validate
from src.utils import get_logger, load_config, resolve

logger = get_logger("predict")


def run(input_path=None, config_path=None):
    cfg = load_config(config_path)
    path = input_path or resolve(cfg["data"]["unlabeled_path"])
    df = pd.read_csv(path)
    ids = df["id"] if "id" in df.columns else pd.Series(range(len(df)), name="id")
    df = df.drop(columns=[c for c in ("id", cfg["target"]) if c in df.columns])

    df = validate(df)
    if cfg["features"]["engineer"]:
        df = engineer_features(df)

    model = joblib.load(resolve(cfg["paths"]["models_dir"]) / "best_model.joblib")
    out = pd.DataFrame({"id": ids, "predicted_price_range": model.predict(df)})
    out_path = resolve(cfg["paths"]["outputs_dir"]) / "predictions.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(out), out_path)
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
