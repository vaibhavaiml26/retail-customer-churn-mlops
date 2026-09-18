"""Model construction, temporal tuning, thresholding and evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config


def build_random_forest_pipeline(**rf_kwargs) -> Pipeline:
    defaults = dict(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1)
    defaults.update(rf_kwargs)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(**defaults)),
    ])


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)),
    ])


def build_xgboost_model(**xgb_kwargs) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    defaults.update(xgb_kwargs)
    return XGBClassifier(**defaults)


def evaluate_fitted_model(model, X_eval, y_eval, label: str = "", threshold: float = 0.5) -> dict:
    """Evaluate an already-fitted model. This function NEVER calls fit()."""
    probs = model.predict_proba(X_eval)[:, 1]
    preds = (probs >= threshold).astype(int)
    mean_prob = float(np.mean(probs))
    actual_rate = float(np.mean(y_eval))
    metrics = {
        "label": label,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_eval, preds)),
        "precision": float(precision_score(y_eval, preds, zero_division=0)),
        "recall": float(recall_score(y_eval, preds, zero_division=0)),
        "f1": float(f1_score(y_eval, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, probs)),
        "pr_auc": float(average_precision_score(y_eval, probs)),
        "brier_score": float(brier_score_loss(y_eval, probs)),
        "mean_churn_probability": mean_prob,
        "actual_churn_rate": actual_rate,
        "calibration_gap": mean_prob - actual_rate,
    }
    print(f"--- {label} ---" if label else "---")
    for key, value in metrics.items():
        if key != "label":
            print(f"{key}: {value:.4f}")
    print(confusion_matrix(y_eval, preds))
    return metrics


def fit_and_evaluate_model(model, X_train, y_train, X_eval, y_eval, label: str = "", threshold: float = 0.5) -> dict:
    model.fit(X_train, y_train)
    return evaluate_fitted_model(model, X_eval, y_eval, label=label, threshold=threshold)


# Backward-compatible name for the development entry point.
def evaluate_model(model, X_train, y_train, X_val, y_val, label: str = "", threshold: float = 0.5) -> dict:
    return fit_and_evaluate_model(model, X_train, y_train, X_val, y_val, label, threshold)


def get_feature_importance(fitted_model, feature_names) -> pd.DataFrame:
    estimator = (
        fitted_model.named_steps["classifier"]
        if hasattr(fitted_model, "named_steps")
        else fitted_model
    )
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def find_best_threshold(fitted_model, X_val, y_val, metric: str = "youden_j", thresholds=None) -> tuple[float, pd.DataFrame]:
    valid = {"f1", "precision", "recall", "youden_j"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}, got {metric!r}")
    thresholds = np.arange(0.05, 0.96, 0.05) if thresholds is None else np.asarray(thresholds)
    probs = fitted_model.predict_proba(X_val)[:, 1]
    rows = []
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": tpr,
            "f1": f1_score(y_val, preds, zero_division=0),
            "fpr": fpr,
            "youden_j": tpr - fpr,
        })
    table = pd.DataFrame(rows)
    best = table.loc[table[metric].idxmax()]
    print(f"Best threshold by {metric}: {best['threshold']:.2f}")
    return float(best["threshold"]), table


def make_forward_chaining_cv(snapshot_ids: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create temporal CV folds from snapshot IDs.

    With training snapshots [2,3,4], folds are 2->3 and [2,3]->4. Rows are
    never randomly moved backward in time. This is the CV behavior production
    forecasting needs; CustomerID grouping alone does not enforce chronology.
    """
    ids = np.asarray(snapshot_ids)
    unique = np.array(sorted(pd.unique(ids)))
    if len(unique) < 2:
        raise ValueError("Temporal CV requires at least two training snapshots.")
    folds = []
    for i in range(1, len(unique)):
        train_snaps = unique[:i]
        val_snap = unique[i]
        train_idx = np.flatnonzero(np.isin(ids, train_snaps))
        val_idx = np.flatnonzero(ids == val_snap)
        folds.append((train_idx, val_idx))
    return folds


def _cv_spec(snapshot_ids_train=None, groups_train=None, cv: int = 3):
    if snapshot_ids_train is not None:
        return make_forward_chaining_cv(snapshot_ids_train), {}
    if groups_train is not None:
        return GroupKFold(n_splits=cv), {"groups": groups_train}
    return cv, {}


def tune_random_forest(
    X_train, y_train, X_val, y_val,
    groups_train=None, snapshot_ids_train=None,
    param_grid: dict | None = None, cv: int = 3,
):
    cv_splitter, fit_kwargs = _cv_spec(snapshot_ids_train, groups_train, cv)
    search = GridSearchCV(
        build_random_forest_pipeline(),
        param_grid or config.RF_PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, **fit_kwargs)
    results = (
        pd.DataFrame(search.cv_results_)[["params", "mean_test_score"]]
        .rename(columns={"mean_test_score": "cv_roc_auc"})
        .sort_values("cv_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 5 Random Forest hyperparameter combinations by CV ROC-AUC:")
    print(results.head())
    best_model = search.best_estimator_
    val_metrics = evaluate_fitted_model(best_model, X_val, y_val, "Best Random Forest (tuned)")
    return best_model, {"params": search.best_params_, "cv_roc_auc": float(search.best_score_), **val_metrics}, results


def tune_xgboost(
    X_train, y_train, X_val, y_val,
    groups_train=None, snapshot_ids_train=None,
    param_grid: dict | None = None, cv: int = 3,
):
    cv_splitter, fit_kwargs = _cv_spec(snapshot_ids_train, groups_train, cv)
    search = GridSearchCV(
        build_xgboost_model(),
        param_grid or config.XGB_PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, **fit_kwargs)
    results = (
        pd.DataFrame(search.cv_results_)[["params", "mean_test_score"]]
        .rename(columns={"mean_test_score": "cv_roc_auc"})
        .sort_values("cv_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 5 XGBoost hyperparameter combinations by CV ROC-AUC:")
    print(results.head())
    best_model = search.best_estimator_
    val_metrics = evaluate_fitted_model(best_model, X_val, y_val, "Best XGBoost (tuned)")
    return best_model, {"params": search.best_params_, "cv_roc_auc": float(search.best_score_), **val_metrics}, results


def build_production_candidate(model_type: str):
    """Build an unfitted quarterly challenger from validated frozen params."""
    if model_type == "random_forest":
        return build_random_forest_pipeline(**config.BEST_RF_PARAMS), dict(config.BEST_RF_PARAMS)
    if model_type == "xgboost":
        return build_xgboost_model(**config.BEST_XGB_PARAMS), dict(config.BEST_XGB_PARAMS)
    raise ValueError(f"Unsupported model_type={model_type!r}")


def temporal_cv_roc_auc(model, X, y, snapshot_ids: pd.Series) -> float:
    """Evaluate fixed hyperparameters with forward-chaining temporal CV."""
    scores = []
    for train_idx, val_idx in make_forward_chaining_cv(snapshot_ids):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], probs))
    return float(np.mean(scores))
