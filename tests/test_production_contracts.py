from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read_required(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.exists():
        pytest.fail(f"Missing required production file: {path}")
    return path.read_text(encoding="utf-8")


def test_delayed_label_contract_counts_only_positive_sales_as_active():
    source = _read_required("azure/evaluate_delayed_performance.py")

    # This is an intentional production contract. Return-only activity must
    # never make an otherwise inactive customer active.
    assert 'positive_sales = outcome_df[outcome_df["Quantity"] > 0]' in source
    assert '"actual_inactive_90d"' in source


def test_scoring_contains_step13a_monitoring_contract():
    source = _read_required("azure/score_azure.py")

    assert "THIS_DOES_NOT_EXIST" in source
    #assert "max_behavioral_feature_psi" in source
    assert "monitoring_record.csv" in source
    assert "prediction_psi" in source


def test_pipeline_uses_persistent_history_locations():
    source = _read_required("azure/submit_pipeline_job.py")

    assert "prediction-history" in source
    assert "performance-history" in source
    assert "alert-history" in source


def test_pipeline_contains_all_four_production_components():
    source = _read_required("azure/submit_pipeline_job.py")

    assert "monthly_score" in source
    assert "delayed_evaluation" in source
    assert "retrain_check" in source
    assert "monitoring_alerts" in source


def test_retrain_child_is_resolved_by_known_node_name():
    source = _read_required("azure/submit_pipeline_job.py")

    # Protect the child-output fix that was required after jobs.list() returned
    # child summaries without populated output metadata.
    assert "retrain_check" in source
    assert "ml_client.jobs.get" in source
