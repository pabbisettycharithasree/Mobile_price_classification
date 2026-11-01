# Lab 10 — FastAPI model serving

Start with `uvicorn api.main:app --reload`. Browse `http://127.0.0.1:8000/docs`. `/health` reports artifact readiness; `/predict` accepts the 20 Mobile Price features and returns a `price_range` prediction. Predictions are appended to `logs/inference/inference_log.jsonl`.
