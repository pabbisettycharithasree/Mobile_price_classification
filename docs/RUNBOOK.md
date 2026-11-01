# Runbook — Labs 10–13

## API
`uvicorn api.main:app --reload`; open `/docs`; check `/health`.

## Deployment
`python src/mlops_pipeline/11_prepare_deployment.py`, then build and run Docker as documented in `lab11_commands/`.

## Monitoring
Run the Lab 12 scripts in the order in `LAB12_MONITORING_DRIFT.md`.

## Integration and tests
`pytest -q` then `python src/mlops_pipeline/13_end_to_end.py`.

If the model artifact paths differ in your existing project, update `api/model_loader.py` to the exact existing paths. Do not retrain just to satisfy the API setup.
