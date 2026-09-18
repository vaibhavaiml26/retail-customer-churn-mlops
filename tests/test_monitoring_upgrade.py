import pandas as pd
import config
import model_registry
from drift_monitoring import (
    compute_feature_drift_report,
    evaluate_delayed_predictions,
    should_promote_challenger,
)
from pipeline_common import append_history_row


def test_temporal_feature_is_reported_separately():
    base = pd.DataFrame({"tenure_days": [10, 20, 30, 40], "recency": [1, 2, 3, 4]})
    current = pd.DataFrame({"tenure_days": [100, 110, 120, 130], "recency": [1, 2, 3, 4]})
    report = compute_feature_drift_report(base, current, expected_temporal_features=["tenure_days"])
    kinds = dict(zip(report["feature"], report["drift_type"]))
    assert kinds["tenure_days"] == "expected_temporal"
    assert kinds["recency"] == "behavioral"


def test_delayed_metrics_include_calibration():
    preds = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "snapshot_date": pd.to_datetime(["2020-01-01"] * 4),
        "churn_probability": [0.1, 0.8, 0.2, 0.9],
        "threshold_used": [0.5] * 4,
    })
    outcomes = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "snapshot_date": pd.to_datetime(["2020-01-01"] * 4),
        "inactive_90d": [0, 1, 0, 1],
    })
    metrics = evaluate_delayed_predictions(preds, outcomes, calibration_bins=2)
    assert metrics["brier_score"] >= 0
    assert "ece" in metrics
    assert len(metrics["calibration_bins"]) == 2
    assert abs(metrics["mean_churn_probability"] - 0.5) < 1e-12


def test_promotion_guardrail_blocks_recall_collapse():
    champion = {"roc_auc": 0.75, "recall": 0.80, "brier_score": 0.20}
    challenger = {"roc_auc": 0.752, "recall": 0.70, "brier_score": 0.20}
    promote, reason = should_promote_challenger(
        challenger,
        champion,
        max_metric_regression=0.005,
        max_recall_regression=0.05,
        max_brier_regression=0.02,
    )
    assert promote is False
    assert "recall regression" in reason


def test_promotion_guardrail_blocks_calibration_degradation():
    champion = {"roc_auc": 0.75, "recall": 0.80, "brier_score": 0.18}
    challenger = {"roc_auc": 0.755, "recall": 0.80, "brier_score": 0.23}
    promote, reason = should_promote_challenger(
        challenger,
        champion,
        max_metric_regression=0.005,
        max_recall_regression=0.05,
        max_brier_regression=0.02,
    )
    assert promote is False
    assert "Brier score" in reason


def test_history_schema_can_evolve(tmp_path):
    path = tmp_path / "history.csv"
    append_history_row(path, {"snapshot_date": "2020-01-01", "roc_auc": 0.75})
    append_history_row(path, {"snapshot_date": "2020-02-01", "roc_auc": 0.76, "brier_score": 0.20})
    hist = pd.read_csv(path)
    assert list(hist.columns) == ["snapshot_date", "roc_auc", "brier_score"]
    assert pd.isna(hist.loc[0, "brier_score"])
    assert hist.loc[1, "brier_score"] == 0.20
