import sys
from common import load_config, ensure_directories, load_train_and_test, project_path, save_json

def main():
    print("[INFO] Lab 7 - Loading mobile price datasets")
    config = load_config()
    ensure_directories(config)
    train_df, test_df = load_train_and_test(config)
    report = {
        "project": "Mobile Price Classification",
        "status": "LOADED",
        "train_path": str(project_path(config["data"]["train_path"])),
        "test_path": str(project_path(config["data"]["test_path"])),
        "train_rows": int(len(train_df)),
        "train_columns": int(len(train_df.columns)),
        "test_rows": int(len(test_df)),
        "test_columns": int(len(test_df.columns)),
        "target_column": config["data"]["target_column"],
        "target_distribution": {
            str(k): int(v) for k, v in train_df[config["data"]["target_column"]].value_counts().sort_index().items()
        },
    }
    save_json("artifacts/data_load_report.json", report)
    print(f"[SUCCESS] Training data: {train_df.shape}; test data: {test_df.shape}")
    print("[SUCCESS] Data loading stage completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Data loading failed: {exc}")
        sys.exit(1)
