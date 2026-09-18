"""Feature/prediction drift, promotion guardrails, and delayed production metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def population_stability_index(baseline: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    baseline = pd.Series(baseline).replace([np.inf, -np.inf], np.nan).dropna()
    current = pd.Series(current).replace([np.inf, -np.inf], np.nan).dropna()
    if baseline.empty or current.empty:
        return float("nan")
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    # Include new production values outside the historical min/max.
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    base_counts = np.histogram(baseline, bins=edges)[0]
    curr_counts = np.histogram(current, bins=edges)[0]
    k = len(base_counts)
    base_pct = (base_counts + 1) / (base_counts.sum() + k)
    curr_pct = (curr_counts + 1) / (curr_counts.sum() + k)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def compute_feature_drift_report(
    X_baseline: pd.DataFrame,
    X_current: pd.DataFrame,
    expected_temporal_features: list[str] | None = None,
) -> pd.DataFrame:
    temporal = set(expected_temporal_features or [])
    rows = []
    for col in X_baseline.columns:
        rows.append({
            "feature": col,
            "psi": population_stability_index(X_baseline[col], X_current[col]),
            "drift_type": "expected_temporal" if col in temporal else "behavioral",
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def compute_prediction_drift(baseline_probs, current_probs) -> float:
    return population_stability_index(pd.Series(baseline_probs), pd.Series(current_probs))


def calibration_by_decile(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Return a robust equal-count calibration table.

    Ranking probabilities first avoids qcut failures when many customers share
    the same probability. Decile 1 is lowest predicted risk; highest decile is
    highest predicted risk.
    """
    frame = pd.DataFrame({"y_true": pd.Series(y_true).astype(int), "y_prob": pd.Series(y_prob).astype(float)})
    if frame.empty:
        return pd.DataFrame(columns=["risk_decile", "n", "mean_predicted_probability", "actual_churn_rate", "absolute_gap"])
    bins = min(int(n_bins), len(frame))
    ranks = frame["y_prob"].rank(method="first")
    frame["risk_decile"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1
    out = (
        frame.groupby("risk_decile", observed=True)
        .agg(
            n=("y_true", "size"),
            mean_predicted_probability=("y_prob", "mean"),
            actual_churn_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    out["risk_decile"] = out["risk_decile"].astype(int)
    out["absolute_gap"] = (out["mean_predicted_probability"] - out["actual_churn_rate"]).abs()
    return out


def expected_calibration_error(calibration_table: pd.DataFrame) -> float:
    if calibration_table.empty:
        return float("nan")
    weights = calibration_table["n"] / calibration_table["n"].sum()
    return float((weights * calibration_table["absolute_gap"]).sum())


def should_promote_challenger(
    challenger_metrics: dict,
    champion_metrics: dict,
    metric: str = "roc_auc",
    max_metric_regression: float = 0.005,
    max_recall_regression: float = 0.05,
    max_brier_regression: float = 0.02,
) -> tuple[bool, str]:
    """Apply primary and safety guardrails to champion/challenger promotion.

    Higher is better for the primary metric and recall. Lower is better for
    Brier score. A candidate must satisfy every configured guardrail.
    """
    primary_delta = float(challenger_metrics[metric]) - float(champion_metrics[metric])
    recall_delta = float(challenger_metrics.get("recall", np.nan)) - float(champion_metrics.get("recall", np.nan))
    brier_delta = float(challenger_metrics.get("brier_score", np.nan)) - float(champion_metrics.get("brier_score", np.nan))

    failures = []
    if primary_delta < -max_metric_regression:
        failures.append(
            f"{metric} regression {primary_delta:+.4f} exceeds allowed -{max_metric_regression:.4f}"
        )
    if np.isfinite(recall_delta) and recall_delta < -max_recall_regression:
        failures.append(
            f"recall regression {recall_delta:+.4f} exceeds allowed -{max_recall_regression:.4f}"
        )
    if np.isfinite(brier_delta) and brier_delta > max_brier_regression:
        failures.append(
            f"Brier score worsened by {brier_delta:+.4f}, exceeding allowed +{max_brier_regression:.4f}"
        )

    headline = (
        f"Challenger {metric}={challenger_metrics[metric]:.4f} vs champion {champion_metrics[metric]:.4f} "
        f"(delta={primary_delta:+.4f}); recall delta={recall_delta:+.4f}; "
        f"Brier delta={brier_delta:+.4f}."
    )
    if failures:
        return False, headline + " BLOCKED: " + "; ".join(failures) + "."

    if primary_delta >= 0:
        primary_text = f"{metric} improved by {primary_delta:+.4f}"
    else:
        primary_text = f"{metric} is within allowed regression ({primary_delta:+.4f})"
    return True, headline + f" PROMOTE: {primary_text}; secondary guardrails passed."


def evaluate_delayed_predictions(
    past_predictions: pd.DataFrame,
    actual_outcomes: pd.DataFrame,
    outcome_col: str = "inactive_90d",
    calibration_bins: int = 10,
) -> dict:
    keys = ["CustomerID", "snapshot_date"]
    joined = past_predictions.merge(actual_outcomes, on=keys, how="inner", validate="one_to_one")
    if joined.empty:
        return {"n_resolved": 0, "note": "no overlapping customers to evaluate", "calibration_bins": []}

    y_true = joined[outcome_col].astype(int)
    y_prob = joined["churn_probability"].astype(float)
    thresholds = joined["threshold_used"].astype(float) if "threshold_used" in joined else pd.Series(0.5, index=joined.index)
    y_pred = (y_prob >= thresholds).astype(int)

    calibration = calibration_by_decile(y_true, y_prob, n_bins=calibration_bins)
    mean_prob = float(y_prob.mean())
    actual_rate = float(y_true.mean())
    result = {
        "n_resolved": int(len(joined)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "actual_churn_rate": actual_rate,
        "mean_churn_probability": mean_prob,
        "predicted_churn_rate": float(y_pred.mean()),
        "mean_threshold_used": float(thresholds.mean()),
        "calibration_gap": mean_prob - actual_rate,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(calibration),
        "calibration_bins": calibration.to_dict("records"),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float("nan")
    return result
