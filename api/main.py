
import time

from fastapi import FastAPI, HTTPException

from .schemas import MobileFeatures, PredictionResponse
from .model_loader import load_artifacts
from .preprocessing import prepare_input
from .monitoring_logger import log_prediction


app = FastAPI(
    title="Mobile Price Classification API",
    version="1.0.0"
)

_model = None
_preprocessor = None

MODEL_VERSION = "local-artifact"


def artifacts():
    global _model, _preprocessor

    if _model is None or _preprocessor is None:
        _model, _preprocessor = load_artifacts()

    return _model, _preprocessor


@app.get("/health")
def health():
    try:
        artifacts()
        return {
            "status": "healthy",
            "model_loaded": True
        }
    except Exception as exc:
        return {
            "status": "not_ready",
            "model_loaded": False,
            "detail": str(exc)
        }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MobileFeatures):
    start_time = time.perf_counter()

    try:
        model, preprocessor = artifacts()

        frame = prepare_input(payload.model_dump())
        transformed = preprocessor.transform(frame)

        pred = int(model.predict(transformed)[0])

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        log_prediction(
            payload.model_dump(),
            pred,
            MODEL_VERSION,
            elapsed_ms
        )

        return PredictionResponse(
            predicted_price_range=pred,
            model_version=MODEL_VERSION
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )
