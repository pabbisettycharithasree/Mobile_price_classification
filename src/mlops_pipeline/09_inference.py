import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
from common import load_config, project_path, configure_mlflow, expected_feature_columns

def main():
    print("[INFO] Lab 7 - Generating predictions for the unlabeled test.csv")
    config = load_config()
    configure_mlflow(config)

    model_path = project_path(config["paths"]["model"])
    preprocessor_path = project_path(config["paths"]["preprocessor"])
    test_path = project_path(config["data"]["test_path"])
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Model or preprocessor artifact is missing. Run training first.")

    test_df = pd.read_csv(test_path)
    features = expected_feature_columns(config)
    X_test = test_df[features].copy()

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    X_test_transformed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_transformed)
    probabilities = model.predict_proba(X_test_transformed)

    output = test_df.copy()
    output["predicted_price_range"] = predictions.astype(int)
    for idx, class_label in enumerate(model.classes_):
        output[f"probability_class_{int(class_label)}"] = np.round(probabilities[:, idx], 6)

    output_path = project_path(config["paths"]["inference_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"[SUCCESS] Generated {len(output)} predictions.")
    print(f"[SUCCESS] Prediction file saved to: {output_path}")
    print("[INFO] The supplied test.csv has no ground-truth labels; no test accuracy is reported.")
    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        sys.exit(1)
