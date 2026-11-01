# Lab 7 — Mobile Price Classification Pipeline Engineering

This adapts the reference churn project's Lab 7 stages to the mobile-price dataset.

## Stages

```text
01 Load Data
02 Raw Data Validation
03 Preprocessing
04 Processed Output Validation
05 Training + MLflow + Model Registry
06 Model Evaluation
08 Automated Model Lifecycle
09 Production Inference
```

Run from the project root:

```powershell
python .\pipelines\run_lab7_pipeline.py
```

The labeled `train.csv` is split into stratified training and validation subsets. The separate `test.csv` has no `price_range` target, so it is used only for inference; no test-set accuracy is claimed.

Metrics include accuracy, macro precision, macro recall and macro F1. The model and preprocessing transformer are saved separately. MLflow tracks parameters/metrics and registers the candidate model. The lifecycle mirrors the reference's `None -> Staging -> Production/Archived` flow.

Generated outputs include processed arrays, model artifacts, MLflow runs, evaluation/lifecycle reports, test predictions and a pipeline run log.
