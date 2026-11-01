"""Lab 5: validate the processed train/test outputs."""
import pandas as pd
from src.utils import load_config, resolve


def run(config_path=None):
    cfg = load_config(config_path)
    d = resolve(cfg["data"]["processed_dir"])
    target = cfg["target"]
    train_path, test_path = d / "train.csv", d / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed train.csv/test.csv were not generated")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for name, df in [("train", train), ("test", test)]:
        if target not in df.columns:
            raise ValueError(f"Processed {name} data has no target column")
        if df.empty:
            raise ValueError(f"Processed {name} data is empty")
        if df.isna().any().any():
            raise ValueError(f"Processed {name} data contains missing values")
    if set(train.columns) != set(test.columns):
        raise ValueError("Processed train/test schemas differ")
    print(f"[SUCCESS] Processed outputs valid: train={train.shape}, test={test.shape}")
    return True


if __name__ == "__main__":
    run()
