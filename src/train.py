"""Train several classifiers with cross-validated grid search and keep the best.

Usage:  python -m src.train
Output: models/best_model.joblib, models/<name>.joblib, outputs/model_comparison.csv
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import get_logger, load_config, resolve

logger = get_logger("train")


def build_pipelines(seed: int) -> dict:
    """Scale-sensitive models get a StandardScaler; tree models don't need one."""
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "logistic_regression": scaled(LogisticRegression(max_iter=2000, random_state=seed)),
        "svc": scaled(SVC(random_state=seed)),
        "knn": scaled(KNeighborsClassifier()),
        "random_forest": Pipeline([("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))]),
        "gradient_boosting": Pipeline([("clf", GradientBoostingClassifier(random_state=seed))]),
    }


def run(config_path=None):
    cfg = load_config(config_path)
    tcfg = cfg["training"]
    target = cfg["target"]

    train = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "train.csv")
    X, y = train.drop(columns=[target]), train[target]
    logger.info("Training on %d rows, %d features", *X.shape)

    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=tcfg["random_state"])
    pipelines = build_pipelines(tcfg["random_state"])
    models_dir = resolve(cfg["paths"]["models_dir"])

    results, best_name, best_score, best_model = [], None, -1.0, None
    for name, grid in tcfg["models"].items():
        param_grid = {f"clf__{k}": v for k, v in grid.items()}
        search = GridSearchCV(pipelines[name], param_grid, cv=cv,
                              scoring=tcfg["scoring"], n_jobs=tcfg["n_jobs"])
        search.fit(X, y)
        logger.info("%-20s CV %s = %.4f  params=%s", name, tcfg["scoring"],
                    search.best_score_, search.best_params_)
        results.append({
            "model": name,
            f"cv_{tcfg['scoring']}": round(search.best_score_, 4),
            "best_params": json.dumps(search.best_params_),
        })
        joblib.dump(search.best_estimator_, models_dir / f"{name}.joblib")
        if search.best_score_ > best_score:
            best_name, best_score, best_model = name, search.best_score_, search.best_estimator_

    comparison = pd.DataFrame(results).sort_values(f"cv_{tcfg['scoring']}", ascending=False)
    comparison.to_csv(resolve(cfg["paths"]["outputs_dir"]) / "model_comparison.csv", index=False)

    joblib.dump(best_model, models_dir / "best_model.joblib")
    (models_dir / "best_model_info.json").write_text(
        json.dumps({"best_model": best_name, "cv_score": round(best_score, 4)}, indent=2))
    logger.info("Best model: %s (CV %.4f) saved to %s", best_name, best_score, models_dir)
    return best_name, comparison


if __name__ == "__main__":
    run()
