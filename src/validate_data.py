"""Lab 5: validate the raw training and prediction datasets."""
import pandas as pd
from src.preprocess import RAW_FEATURES
from src.utils import load_config, resolve


def validate_file(path, target_required=False):
    df = pd.read_csv(path)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if target_required and "price_range" not in df.columns:
        raise ValueError(f"{path} must contain price_range")
    if df[RAW_FEATURES].isna().any().any():
        raise ValueError(f"{path} contains missing feature values")
    if target_required and df["price_range"].isna().any():
        raise ValueError(f"{path} contains missing target values")
    return df


def run(config_path=None):
    cfg = load_config(config_path)
    train_path = resolve(cfg["data"]["raw_path"])
    train = validate_file(train_path, target_required=True)
    if set(train["price_range"].unique()) - {0,1,2,3}:
        raise ValueError("price_range must contain only 0,1,2,3")
    unlabeled = resolve(cfg["data"]["unlabeled_path"])
    if unlabeled.exists():
        validate_file(unlabeled, target_required=False)
        print(f"[SUCCESS] Validated {unlabeled}")
    print(f"[SUCCESS] Validated training data: {train.shape}")
    return True


if __name__ == "__main__":
    run()
