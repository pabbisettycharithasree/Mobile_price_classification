# Lab 9 — Mobile Price Pipeline Failure Handling and Recovery

## Goal

Inject controlled failures into the Lab 7-8 pipeline and verify that validation stops unsafe execution or that the model can be restored from the MLflow Production registry.

## Run locally

Run the normal Lab 8 pipeline first so processed data, a registered Production model and reports exist:

```powershell
python .\pipelines\run_lab8_pipeline.py
python .\src\mlops_pipeline\10_failure_injection.py
```

## Scenarios

| Scenario | Expected behavior |
|---|---|
| Missing `data/raw/train.csv` | Raw validation exits non-zero (HALT) |
| Invalid numeric feature value | Raw validation exits non-zero (HALT) |
| Missing feature column | Schema validation exits non-zero (HALT) |
| NaN injected into processed training array | Processed-output validation exits non-zero (HALT) |
| Local model artifact removed | Evaluation detects failure; script restores the MLflow Production model and retries evaluation (RECOVER) |

The failure script backs up each file and restores it in `finally` blocks. Still, close other programs that might edit the CSVs while tests are running, and do not interrupt the process during a test.

## Report

The result is written to `reports/failure_report.json`. `lab9.repetitions` controls the number of repetitions in `config_lab7.yaml` (default 1).

The tests require a successful Lab 8 run and a Production model registered in the configured MLflow store. If there is no Production version, the recovery scenario is reported as failed/not recoverable rather than fabricating a successful recovery.

## GitHub Actions

Workflow: `.github/workflows/lab9_failure_tests.yml`. It is manually triggered, builds/registers the normal model first, runs the failure scenarios, and uploads reports as workflow artifacts.
