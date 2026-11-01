# Lab 11 — Docker and Kubernetes

Build the image from the project root with `docker build -t mobile-price-api:latest .`. Start locally with `docker run --rm -p 8000:8000 mobile-price-api:latest`. Apply `k8s/deployment.yaml` and `k8s/service.yaml` after the image is available to the cluster. Confirm `/health` and `/docs` before declaring deployment successful.

The model and preprocessor are project artifacts. Prepare them with `11_prepare_deployment.py`; do not substitute churn model files.
