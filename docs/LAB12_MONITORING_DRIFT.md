# Lab 12 — Monitoring and drift

From project root, run:

1. `python src/mlops_pipeline/12_create_evolved_dataset.py`
2. `python src/mlops_pipeline/12_data_drift.py`
3. `python src/mlops_pipeline/12_drift_analysis.py`
4. `python src/mlops_pipeline/12_prediction_monitoring.py`
5. `python src/mlops_pipeline/12_monitor_trigger.py`

The PSI threshold is 0.2. Synthetic evolved samples are generated separately under `data/monitoring/drift/`; `data/raw/train.csv` is not changed. Retraining and registry scripts are decision-support stubs and do not silently replace or promote the production model.
