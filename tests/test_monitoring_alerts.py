from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERT_SCRIPT = PROJECT_ROOT / "azure" / "evaluate_monitoring_alerts.py"


def _load_alert_module():
    if not ALERT_SCRIPT.exists():
        pytest.fail(f"Missing required production file: {ALERT_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "retail_churn_monitoring_alerts",
        ALERT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prediction_psi_threshold_boundaries():
    m = _load_alert_module()

    ok = m._higher_is_bad_signal(
        name="prediction_psi",
        current=0.099,
        warning=0.10,
        critical=0.25,
        detail="test",
    )
    warning = m._higher_is_bad_signal(
        name="prediction_psi",
        current=0.10,
        warning=0.10,
        critical=0.25,
        detail="test",
    )
    critical = m._higher_is_bad_signal(
        name="prediction_psi",
        current=0.25,
        warning=0.10,
        critical=0.25,
        detail="test",
    )

    assert ok["status"] == "OK"
    assert warning["status"] == "WARNING"
    assert critical["status"] == "CRITICAL"


def test_metric_regression_direction_is_correct():
    m = _load_alert_module()

    # ROC-AUC: lower current value is worse.
    roc = m._regression_signal(
        name="roc",
        reference=0.75,
        current=0.74,
        warning_regression=0.005,
        critical_regression=0.02,
        metric_direction="higher_is_better",
        detail="test",
    )
    assert roc["status"] == "WARNING"
    assert roc["regression"] == pytest.approx(0.01)

    # Brier: higher current value is worse.
    brier = m._regression_signal(
        name="brier",
        reference=0.20,
        current=0.23,
        warning_regression=0.02,
        critical_regression=0.04,
        metric_direction="lower_is_better",
        detail="test",
    )
    assert brier["status"] == "WARNING"
    assert brier["regression"] == pytest.approx(0.03)


def test_reference_metrics_prefer_heldout_test():
    m = _load_alert_module()

    metadata = {
        "validation_metrics": {
            "roc_auc": 0.77,
            "recall": 0.80,
            "brier_score": 0.21,
        },
        "test_metrics": {
            "roc_auc": 0.75,
            "recall": 0.78,
            "brier_score": 0.22,
        },
    }

    source, metrics = m._resolve_reference_metrics(metadata)

    assert source == "test_metrics"
    assert metrics["roc_auc"] == 0.75


def test_alert_main_uses_behavioral_psi_not_tenure(tmp_path):
    m = _load_alert_module()

    score_dir = tmp_path / "score"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "alerts"

    score_dir.mkdir()
    model_dir.mkdir()

    # Overall max PSI is intentionally enormous because tenure_days drifted.
    # Behavioral PSI remains below warning threshold.
    (score_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "snapshot_date": "2011-09-01",
                "model_lineage": "azureml:retail-churn-xgboost:2",
                "prediction_psi": 0.08,
                "max_feature_psi": 1.80,
                "max_behavioral_feature_psi": 0.09,
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    "roc_auc": 0.7517,
                    "recall": 0.8444,
                    "brier_score": 0.2048,
                }
            }
        ),
        encoding="utf-8",
    )

    result = m.main(
        score_data=str(score_dir),
        model_path=str(model_dir),
        model_asset_ref="azureml:retail-churn-xgboost:2",
        alert_output=str(output_dir),
        performance_data=None,
        prediction_psi_warning=0.10,
        prediction_psi_critical=0.25,
        feature_psi_warning=0.10,
        feature_psi_critical=0.25,
        max_roc_auc_drop=0.005,
        max_recall_drop=0.05,
        max_brier_worsening=0.02,
        critical_multiplier=2.0,
    )

    assert result["overall_status"] == "OK"
    assert result["max_feature_psi_including_tenure"] == pytest.approx(1.80)
    assert result["max_behavioral_feature_psi"] == pytest.approx(0.09)

    assert (output_dir / "alert_summary.json").exists()
    assert (output_dir / "alert_record.csv").exists()
    assert (output_dir / "alert_signals.csv").exists()


def test_delayed_performance_can_raise_critical_alert(tmp_path):
    m = _load_alert_module()

    score_dir = tmp_path / "score"
    model_dir = tmp_path / "model"
    performance_dir = tmp_path / "performance"
    output_dir = tmp_path / "alerts"

    score_dir.mkdir()
    model_dir.mkdir()
    performance_dir.mkdir()

    (score_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "snapshot_date": "2011-12-01",
                "model_lineage": "azureml:retail-churn-xgboost:2",
                "prediction_psi": 0.08,
                "max_feature_psi": 1.7,
                "max_behavioral_feature_psi": 0.07,
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    "roc_auc": 0.75,
                    "recall": 0.84,
                    "brier_score": 0.20,
                }
            }
        ),
        encoding="utf-8",
    )

    # Deliberately poor delayed performance.
    (performance_dir / "performance_metrics_2011-09-01.json").write_text(
        json.dumps(
            {
                "evaluation_skipped": False,
                "prediction_snapshot": "2011-09-01",
                "model_lineage": "azureml:retail-churn-xgboost:2",
                "roc_auc": 0.72,
                "recall": 0.70,
                "brier_score": 0.25,
            }
        ),
        encoding="utf-8",
    )

    result = m.main(
        score_data=str(score_dir),
        model_path=str(model_dir),
        model_asset_ref="azureml:retail-churn-xgboost:2",
        alert_output=str(output_dir),
        performance_data=str(performance_dir),
        prediction_psi_warning=0.10,
        prediction_psi_critical=0.25,
        feature_psi_warning=0.10,
        feature_psi_critical=0.25,
        max_roc_auc_drop=0.005,
        max_recall_drop=0.05,
        max_brier_worsening=0.02,
        critical_multiplier=2.0,
    )

    assert result["overall_status"] == "CRITICAL"
    assert "delayed_roc_auc_regression" in result["critical_signals"]
    assert "delayed_recall_regression" in result["critical_signals"]
    assert "delayed_brier_worsening" in result["critical_signals"]


def test_different_model_lineage_is_not_compared(tmp_path):
    m = _load_alert_module()

    score_dir = tmp_path / "score"
    model_dir = tmp_path / "model"
    performance_dir = tmp_path / "performance"
    output_dir = tmp_path / "alerts"

    score_dir.mkdir()
    model_dir.mkdir()
    performance_dir.mkdir()

    (score_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "snapshot_date": "2011-12-01",
                "model_lineage": "azureml:retail-churn-xgboost:3",
                "prediction_psi": 0.05,
                "max_feature_psi": 1.2,
                "max_behavioral_feature_psi": 0.05,
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    "roc_auc": 0.80,
                    "recall": 0.90,
                    "brier_score": 0.15,
                }
            }
        ),
        encoding="utf-8",
    )

    # Old prediction was made by v2 but current model supplied is v3.
    (performance_dir / "performance_metrics_2011-09-01.json").write_text(
        json.dumps(
            {
                "evaluation_skipped": False,
                "prediction_snapshot": "2011-09-01",
                "model_lineage": "azureml:retail-churn-xgboost:2",
                "roc_auc": 0.60,
                "recall": 0.50,
                "brier_score": 0.35,
            }
        ),
        encoding="utf-8",
    )

    result = m.main(
        score_data=str(score_dir),
        model_path=str(model_dir),
        model_asset_ref="azureml:retail-churn-xgboost:3",
        alert_output=str(output_dir),
        performance_data=str(performance_dir),
        prediction_psi_warning=0.10,
        prediction_psi_critical=0.25,
        feature_psi_warning=0.10,
        feature_psi_critical=0.25,
        max_roc_auc_drop=0.005,
        max_recall_drop=0.05,
        max_brier_worsening=0.02,
        critical_multiplier=2.0,
    )

    assert result["performance"]["reference_comparable"] is False

    status_by_signal = {
        signal["signal"]: signal["status"]
        for signal in result["signals"]
    }
    assert status_by_signal["delayed_performance_reference"] == "NOT_COMPARABLE"
