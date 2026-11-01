# Lab 8 — Pipeline Automation and CI Workflows

## Goal

Automate the existing Lab 7 mobile-price pipeline with GitHub Actions and add the quality gate from the reference project. CI does not define a second training implementation; it invokes the same repository pipeline.

## Local run

```powershell
python .\pipelines\run_lab8_pipeline.py
```

The Lab 8 run inserts `07_quality_gate.py` between evaluation and lifecycle management:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 Quality Gate -> 08 -> 09
```

The gate checks macro F1 against `quality_gate.minimum_value` in `config_lab7.yaml`. The default is 0.80 and should be changed only after reviewing validation performance and project requirements. A failed gate stops the pipeline before model lifecycle promotion.

## GitHub Actions

Workflow: `.github/workflows/lab8_ci.yml`.

Triggers: pushes and pull requests affecting the Lab 7-8 configuration, source, pipeline, requirements or datasets, plus manual `workflow_dispatch`.

After pushing the project to GitHub, open **Actions -> Lab 8 - Mobile Price MLOps CI -> Run workflow** to test manually. CI needs `data/raw/train.csv` and `data/raw/test.csv` available in the checkout. Do not commit credentials, virtual environments, local MLflow databases or generated model binaries.

## MLflow in CI

For persistent experiment history across ephemeral GitHub runners, configure a repository Actions secret named `MLFLOW_TRACKING_URI` pointing to a reachable MLflow Tracking Server. A local `sqlite:///mlflow.db` is suitable for local development, but is not shared persistent storage across GitHub-hosted runners. See `MLFLOW_CI_SETUP.md`.
