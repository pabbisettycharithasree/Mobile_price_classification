"""Data loading, validation, feature engineering and train/test split.

Usage:  python -m src.preprocess
Output: data/processed/train.csv, data/processed/test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, load_config, resolve

logger = get_logger("preprocess")

RAW_FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory",
    "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", "sc_h",
    "sc_w", "talk_time", "three_g", "touch_screen", "wifi",
]


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %s -> shape %s", path, df.shape)
    return df


def validate(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Check schema, drop duplicates, report missing values."""
    missing_cols = [c for c in RAW_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        if df[target].isna().any():
            raise ValueError("Target contains missing values")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        logger.warning("%d missing values found - filling numeric columns with median", n_missing)
        df = df.fillna(df.median(numeric_only=True))

    n_dup = int(df.duplicated().sum())
    if n_dup:
        logger.warning("Dropping %d duplicate rows", n_dup)
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Safe to call on labelled and unlabelled data."""
    df = df.copy()
    df["px_area"] = df["px_height"] * df["px_width"]                # total pixels
    df["screen_area"] = df["sc_h"] * df["sc_w"]                     # physical screen size (cm^2)
    df["px_density"] = df["px_area"] / (df["screen_area"] + 1)      # +1: sc_w/sc_h can be 0
    df["camera_total"] = df["fc"] + df["pc"]                        # front + primary camera MP
    df["ram_per_core"] = df["ram"] / df["n_cores"].clip(lower=1)
    return df


def split_and_save(df: pd.DataFrame, cfg: dict):
    target = cfg["target"]
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )
    out_dir = resolve(cfg["data"]["processed_dir"])
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    logger.info("Saved processed data: train %s, test %s -> %s", train.shape, test.shape, out_dir)
    return train, test


def run(config_path=None):
    cfg = load_config(config_path)
    df = load_raw(resolve(cfg["data"]["raw_path"]))
    df = validate(df, cfg["target"])
    if cfg["features"]["engineer"]:
        df = engineer_features(df)
    return split_and_save(df, cfg)


if __name__ == "__main__":
    run()
