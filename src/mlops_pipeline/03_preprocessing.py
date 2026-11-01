import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import load_config, ensure_directories, load_train_and_test, project_path, expected_feature_columns

def main():
    print("[INFO] Lab 7 - Preprocessing and train/validation split")
    config = load_config()
    ensure_directories(config)
    train_df, competition_test_df = load_train_and_test(config)

    target = config["data"]["target_column"]
    id_col = config["data"].get("id_column", "id")
    features = expected_feature_columns(config)

    X = train_df[features].copy()
    y = train_df[target].astype(int).copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=float(config["split"]["validation_size"]),
        random_state=int(config["split"]["random_state"]),
        stratify=y
    )

    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_final = preprocessor.fit_transform(X_train)
    X_valid_final = preprocessor.transform(X_valid)
    X_competition_test = preprocessor.transform(competition_test_df[features].copy())

    processed_dir = project_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_valid_final.npy", X_valid_final)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(processed_dir / "y_valid.npy", y_valid.to_numpy(dtype=np.int64))
    np.save(processed_dir / "X_test_final.npy", X_competition_test)

    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_valid.to_csv(processed_dir / "X_valid_raw.csv", index=False)
    competition_test_df.to_csv(processed_dir / "competition_test_raw.csv", index=False)

    preprocessor_path = project_path(config["paths"]["preprocessor"])
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "dataset_name": "Mobile Price Classification",
        "target_column": target,
        "feature_columns": features,
        "train_shape_raw": list(X_train.shape),
        "validation_shape_raw": list(X_valid.shape),
        "competition_test_shape_raw": list(competition_test_df[features].shape),
        "train_shape_transformed": list(X_train_final.shape),
        "validation_shape_transformed": list(X_valid_final.shape),
        "competition_test_shape_transformed": list(X_competition_test.shape),
        "validation_size": float(config["split"]["validation_size"]),
        "random_state": int(config["split"]["random_state"]),
        "split_strategy": "stratified train/validation split from labeled train.csv",
        "external_test_has_labels": False
    }
    with (processed_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Train matrix: {X_train_final.shape}")
    print(f"[SUCCESS] Validation matrix: {X_valid_final.shape}")
    print(f"[SUCCESS] Unlabeled test matrix: {X_competition_test.shape}")
    print(f"[SUCCESS] Preprocessor saved: {preprocessor_path}")
    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:
        print(f"[ERROR] Preprocessing failed: {exc}")
        sys.exit(1)
