"""Evaluate the best model on the held-out test split.

Usage:  python -m src.evaluate
Output: reports/metrics.json, reports/classification_report.txt, reports/summary.md,
        outputs/confusion_matrix.png, outputs/feature_importance.png
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.utils import get_logger, load_config, resolve

logger = get_logger("evaluate")


def plot_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, X, y, path, top_n=15, seed=42):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=seed, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(top_n)[::-1].plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} features (permutation importance)")
    ax.set_xlabel("Mean drop in accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp


def run(config_path=None):
    cfg = load_config(config_path)
    target = cfg["target"]
    models_dir = resolve(cfg["paths"]["models_dir"])
    outputs = resolve(cfg["paths"]["outputs_dir"])
    reports = resolve(cfg["paths"]["reports_dir"])

    model = joblib.load(models_dir / "best_model.joblib")
    info = json.loads((models_dir / "best_model_info.json").read_text())
    test = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "test.csv")
    X_test, y_test = test.drop(columns=[target]), test[target]

    y_pred = model.predict(X_test)
    metrics = {
        "model": info["best_model"],
        "cv_score": info["cv_score"],
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
    }
    report = classification_report(y_test, y_pred, target_names=cfg["class_names"])
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report)

    plot_confusion_matrix(y_test, y_pred, cfg["class_names"], outputs / "confusion_matrix.png")
    imp = plot_feature_importance(model, X_test, y_test, outputs / "feature_importance.png",
                                  seed=cfg["training"]["random_state"])

    top5 = "\n".join(f"- `{k}`: {v:.4f}" for k, v in imp.head(5).items())
    summary = f"""# Model Evaluation Summary

**Best model:** `{metrics['model']}`  |  **CV accuracy:** {metrics['cv_score']}

| Metric | Test value |
|---|---|
| Accuracy | {metrics['test_accuracy']} |
| Precision (macro) | {metrics['precision_macro']} |
| Recall (macro) | {metrics['recall_macro']} |
| F1 (macro) | {metrics['f1_macro']} |

## Classification report
```
{report}
```

## Top 5 features
{top5}

Figures: `outputs/confusion_matrix.png`, `outputs/feature_importance.png`
"""
    (reports / "summary.md").write_text(summary)
    logger.info("Test metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    run()
