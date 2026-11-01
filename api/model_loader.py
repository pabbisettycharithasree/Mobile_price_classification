from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mobile_price_random_forest.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "mobile_price_preprocessor.pkl"

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run Lab 9 and confirm model artifacts are present.")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}. Run the existing training pipeline first.")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)
