# Mobile Price Classification MLOps architecture (Labs 10–13)

Labs 1–9 remain the existing training, validation, registry, inference and failure-handling foundation. Lab 10 exposes the saved model through FastAPI. Lab 11 packages the API for Docker/Kubernetes. Lab 12 monitors feature and prediction distributions using API inference logs and generated drift samples. Lab 13 runs the deployment/monitoring checks without rerunning Labs 1–9.

The model contract is the 20 Mobile Price input columns and `price_range` output. The API expects `models/mobile_price_random_forest.pkl` and `models/mobile_price_preprocessor.pkl` from the existing project.
