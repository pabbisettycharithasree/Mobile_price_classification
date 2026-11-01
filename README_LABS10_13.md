# Mobile Price Classification — Labs 10–13 overlay

This is an additive overlay based on the supplied Churn Prediction project layout. It is designed to merge into the existing Mobile Price project after Labs 1–9.

## Preserve completed work
Do not delete or replace `src/mlops_pipeline/01_*` through `10_*`, `pipelines/run_lab7_pipeline.py`, `pipelines/run_lab8_pipeline.py`, existing data, models, MLflow database/runs, reports or outputs. The ZIP contains only new Labs 10–13 paths plus shared-folder placeholders/documentation.

## Prerequisites
Existing model files expected by default: `models/mobile_price_random_forest.pkl` and `models/mobile_price_preprocessor.pkl`. If your actual filenames differ, edit `api/model_loader.py` only to point to your existing artifacts.

## Run
- API: `uvicorn api.main:app --reload`
- Tests: `pytest -q`
- Deployment prep: `python src/mlops_pipeline/11_prepare_deployment.py`
- Monitoring: see `docs/LAB12_MONITORING_DRIFT.md`
- Integration: `python src/mlops_pipeline/13_end_to_end.py`

This is a project scaffold, not a claim that Docker/Kubernetes have been run in your environment. Model binaries and generated reports are not fabricated or copied from churn.
