"""Lab 4: validate that the saved model gives deterministic predictions."""
import hashlib
import joblib
import pandas as pd

from src.utils import load_config, resolve


def run(config_path=None):
    cfg = load_config(config_path)
    model = joblib.load(resolve(cfg["paths"]["models_dir"]) / "best_model.joblib")
    test = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "test.csv")
    X = test.drop(columns=[cfg["target"]])
    p1 = model.predict(X)
    p2 = model.predict(X)
    same = (p1 == p2).all()
    digest = hashlib.sha256(p1.tobytes()).hexdigest()
    if not same:
        raise RuntimeError("Reproducibility check failed: predictions changed between runs")
    print("[SUCCESS] Reproducibility check passed")
    print(f"[INFO] Prediction SHA-256: {digest}")
    return True


if __name__ == "__main__":
    run()
