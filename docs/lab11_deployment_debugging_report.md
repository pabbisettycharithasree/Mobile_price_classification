# Lab 11 deployment debugging checklist

- Verify model and preprocessor exist under `models/`.
- Verify API feature names and preprocessing order match the Lab 9 training contract.
- Inspect `docker logs` and `/health` before testing `/predict`.
- Ensure the Kubernetes image name and pull policy match the locally built image.
- Never overwrite Labs 1–9 outputs during deployment troubleshooting.
