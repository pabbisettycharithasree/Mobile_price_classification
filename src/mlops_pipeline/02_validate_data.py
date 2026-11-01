import sys
from common import load_config, load_train_and_test, validate_train_test_schema, save_json

def main():
    print("[INFO] Lab 7 - Raw data validation")
    config = load_config()
    train_df, test_df = load_train_and_test(config)
    passed, errors = validate_train_test_schema(train_df, test_df, config)
    report = {
        "stage": "raw_data_validation",
        "status": "PASSED" if passed else "FAILED",
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "errors": errors,
    }
    save_json("artifacts/raw_data_validation_report.json", report)
    if not passed:
        for error in errors:
            print(f"[ERROR] {error}")
        return False
    print("[SUCCESS] Training and test schemas are valid.")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
