"""Evaluate retail-churn monitoring signals and emit a consolidated alert summary.

This component is intentionally non-blocking: WARNING/CRITICAL monitoring states
are persisted as artifacts but do not fail the Azure ML pipeline.

Immediate signals
-----------------
* prediction PSI
* max behavioral feature PSI (excluding tenure_days)

Delayed performance signals, when a matured evaluation is available
-------------------------------------------------------------------
* ROC-AUC regression versus the model's reference test/validation metric
* recall regression versus the model's reference test/validation metric
* Brier-score worsening versus the model's reference test/validation metric

Outputs
-------
* alert_summary.json
* alert_record.csv
* alert_signals.csv
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STATUS_RANK = {
    "OK": 0,
    "NOT_AVAILABLE": 0,
    "NOT_COMPARABLE": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def _find_exactly_one(root: Path, filename: str) -> Path:
    candidates = list(root.rglob(filename)) if root.is_dir() else []
    if root.is_file() and root.name == filename:
        candidates = [root]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {filename!r} under {root}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _find_zero_or_one_glob(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = list(root.rglob(pattern)) if root.is_dir() else []
    if root.is_file() and root.match(pattern):
        candidates = [root]
    if len(candidates) == 0:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"Expected zero or one file matching {pattern!r} under {root}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _higher_is_bad_signal(
    *,
    name: str,
    current: float | None,
    warning: float,
    critical: float,
    detail: str,
) -> dict:
    if current is None:
        status = "NOT_AVAILABLE"
    elif current >= critical:
        status = "CRITICAL"
    elif current >= warning:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "signal": name,
        "status": status,
        "current_value": current,
        "warning_threshold": float(warning),
        "critical_threshold": float(critical),
        "direction": "higher_is_worse",
        "detail": detail,
    }


def _regression_signal(
    *,
    name: str,
    reference: float | None,
    current: float | None,
    warning_regression: float,
    critical_regression: float,
    metric_direction: str,
    detail: str,
) -> dict:
    if reference is None or current is None:
        return {
            "signal": name,
            "status": "NOT_AVAILABLE",
            "reference_value": reference,
            "current_value": current,
            "regression": None,
            "warning_regression": float(warning_regression),
            "critical_regression": float(critical_regression),
            "detail": detail,
        }

    if metric_direction == "higher_is_better":
        regression = float(reference - current)
    elif metric_direction == "lower_is_better":
        regression = float(current - reference)
    else:
        raise ValueError(f"Unsupported metric_direction: {metric_direction}")

    if regression >= critical_regression:
        status = "CRITICAL"
    elif regression >= warning_regression:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "signal": name,
        "status": status,
        "reference_value": float(reference),
        "current_value": float(current),
        "regression": regression,
        "warning_regression": float(warning_regression),
        "critical_regression": float(critical_regression),
        "direction": metric_direction,
        "detail": detail,
    }


def _get_behavioral_feature_psi(score_root: Path, summary: dict) -> float | None:
    # Step 13A writes this directly. Keep a fallback for older scoring outputs.
    direct = _numeric_or_none(summary.get("max_behavioral_feature_psi"))
    if direct is not None:
        return direct

    drift_file = _find_zero_or_one_glob(score_root, "feature_drift.csv")
    if drift_file is None:
        return None

    drift = pd.read_csv(drift_file)
    if drift.empty or "psi" not in drift.columns or "feature" not in drift.columns:
        return None

    behavioral = drift[drift["feature"].astype(str) != "tenure_days"].copy()
    if behavioral.empty:
        return None

    return float(pd.to_numeric(behavioral["psi"], errors="coerce").max())


def _resolve_reference_metrics(metadata: dict) -> tuple[str | None, dict]:
    """Prefer held-out test metrics; fall back to validation/metrics."""
    for source in ("test_metrics", "validation_metrics", "metrics"):
        metrics = metadata.get(source)
        if isinstance(metrics, dict) and metrics:
            needed = {"roc_auc", "recall", "brier_score"}
            if any(key in metrics for key in needed):
                return source, metrics
    return None, {}


def _max_status(signals: list[dict]) -> str:
    candidates = [s.get("status", "NOT_AVAILABLE") for s in signals]
    if not candidates:
        return "OK"
    return max(candidates, key=lambda s: STATUS_RANK.get(s, 0))


def main(
    score_data: str,
    model_path: str,
    model_asset_ref: str,
    alert_output: str,
    performance_data: str | None,
    prediction_psi_warning: float,
    prediction_psi_critical: float,
    feature_psi_warning: float,
    feature_psi_critical: float,
    max_roc_auc_drop: float,
    max_recall_drop: float,
    max_brier_worsening: float,
    critical_multiplier: float,
) -> dict:
    score_root = Path(score_data)
    model_root = Path(model_path)
    output_dir = Path(alert_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_file = _find_exactly_one(score_root, "run_summary.json")
    run_summary = _load_json(run_summary_file)

    metadata_file = _find_exactly_one(model_root, "metadata.json")
    model_metadata = _load_json(metadata_file)

    snapshot_date = run_summary.get("snapshot_date")
    prediction_psi = _numeric_or_none(run_summary.get("prediction_psi"))
    max_feature_psi = _numeric_or_none(run_summary.get("max_feature_psi"))
    max_behavioral_feature_psi = _get_behavioral_feature_psi(
        score_root, run_summary
    )

    signals: list[dict] = []

    signals.append(
        _higher_is_bad_signal(
            name="prediction_psi",
            current=prediction_psi,
            warning=prediction_psi_warning,
            critical=prediction_psi_critical,
            detail="PSI of current churn probabilities versus model reference probabilities.",
        )
    )
    signals.append(
        _higher_is_bad_signal(
            name="max_behavioral_feature_psi",
            current=max_behavioral_feature_psi,
            warning=feature_psi_warning,
            critical=feature_psi_critical,
            detail=(
                "Maximum feature PSI excluding tenure_days, which is tracked "
                "separately because calendar-time drift is structurally expected."
            ),
        )
    )

    performance_summary: dict[str, Any] = {
        "available": False,
        "reference_comparable": False,
    }

    if performance_data:
        performance_root = Path(performance_data)
        metrics_file = _find_zero_or_one_glob(
            performance_root, "performance_metrics_*.json"
        )

        if metrics_file is not None:
            performance_metrics = _load_json(metrics_file)
            evaluation_skipped = bool(
                performance_metrics.get("evaluation_skipped", False)
            )

            performance_summary = {
                "available": not evaluation_skipped,
                "evaluation_skipped": evaluation_skipped,
                "reason": performance_metrics.get("reason"),
                "prediction_snapshot": performance_metrics.get(
                    "prediction_snapshot"
                ),
                "model_lineage": performance_metrics.get("model_lineage"),
                "reference_model": model_asset_ref,
                "reference_comparable": False,
            }

            if not evaluation_skipped:
                delayed_model_ref = performance_metrics.get("model_lineage")
                same_model = (
                    delayed_model_ref is None
                    or str(delayed_model_ref) == str(model_asset_ref)
                )
                performance_summary["reference_comparable"] = same_model

                if same_model:
                    reference_source, reference = _resolve_reference_metrics(
                        model_metadata
                    )
                    performance_summary["reference_source"] = reference_source
                    performance_summary["reference_metrics"] = {
                        "roc_auc": _numeric_or_none(reference.get("roc_auc")),
                        "recall": _numeric_or_none(reference.get("recall")),
                        "brier_score": _numeric_or_none(
                            reference.get("brier_score")
                        ),
                    }
                    performance_summary["current_metrics"] = {
                        "roc_auc": _numeric_or_none(
                            performance_metrics.get("roc_auc")
                        ),
                        "recall": _numeric_or_none(
                            performance_metrics.get("recall")
                        ),
                        "brier_score": _numeric_or_none(
                            performance_metrics.get("brier_score")
                        ),
                    }

                    signals.extend(
                        [
                            _regression_signal(
                                name="delayed_roc_auc_regression",
                                reference=_numeric_or_none(
                                    reference.get("roc_auc")
                                ),
                                current=_numeric_or_none(
                                    performance_metrics.get("roc_auc")
                                ),
                                warning_regression=max_roc_auc_drop,
                                critical_regression=(
                                    max_roc_auc_drop * critical_multiplier
                                ),
                                metric_direction="higher_is_better",
                                detail=(
                                    "Delayed production ROC-AUC compared with "
                                    f"registered-model {reference_source} baseline."
                                ),
                            ),
                            _regression_signal(
                                name="delayed_recall_regression",
                                reference=_numeric_or_none(
                                    reference.get("recall")
                                ),
                                current=_numeric_or_none(
                                    performance_metrics.get("recall")
                                ),
                                warning_regression=max_recall_drop,
                                critical_regression=(
                                    max_recall_drop * critical_multiplier
                                ),
                                metric_direction="higher_is_better",
                                detail=(
                                    "Delayed production recall compared with "
                                    f"registered-model {reference_source} baseline."
                                ),
                            ),
                            _regression_signal(
                                name="delayed_brier_worsening",
                                reference=_numeric_or_none(
                                    reference.get("brier_score")
                                ),
                                current=_numeric_or_none(
                                    performance_metrics.get("brier_score")
                                ),
                                warning_regression=max_brier_worsening,
                                critical_regression=(
                                    max_brier_worsening * critical_multiplier
                                ),
                                metric_direction="lower_is_better",
                                detail=(
                                    "Delayed production Brier score compared with "
                                    f"registered-model {reference_source} baseline."
                                ),
                            ),
                        ]
                    )
                else:
                    signals.append(
                        {
                            "signal": "delayed_performance_reference",
                            "status": "NOT_COMPARABLE",
                            "current_model_lineage": delayed_model_ref,
                            "reference_model": model_asset_ref,
                            "detail": (
                                "Delayed predictions were produced by a different "
                                "model version than the model supplied to this alert "
                                "component; performance deltas were not calculated."
                            ),
                        }
                    )

    overall_status = _max_status(signals)
    warning_signals = [
        s["signal"] for s in signals if s.get("status") == "WARNING"
    ]
    critical_signals = [
        s["signal"] for s in signals if s.get("status") == "CRITICAL"
    ]

    created_at = datetime.now(timezone.utc).isoformat()
    alert_summary = {
        "stage": "azure_monitoring_alerts",
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "snapshot_date": snapshot_date,
        "model_lineage": run_summary.get("model_lineage"),
        "model_asset_ref": model_asset_ref,
        "overall_status": overall_status,
        "warning_count": len(warning_signals),
        "critical_count": len(critical_signals),
        "warning_signals": warning_signals,
        "critical_signals": critical_signals,
        "prediction_psi": prediction_psi,
        "max_feature_psi_including_tenure": max_feature_psi,
        "max_behavioral_feature_psi": max_behavioral_feature_psi,
        "performance": performance_summary,
        "thresholds": {
            "prediction_psi_warning": prediction_psi_warning,
            "prediction_psi_critical": prediction_psi_critical,
            "feature_psi_warning": feature_psi_warning,
            "feature_psi_critical": feature_psi_critical,
            "max_roc_auc_drop": max_roc_auc_drop,
            "max_recall_drop": max_recall_drop,
            "max_brier_worsening": max_brier_worsening,
            "critical_multiplier": critical_multiplier,
        },
        "signals": signals,
        "created_at": created_at,
    }

    _write_json(output_dir / "alert_summary.json", alert_summary)

    flat_record = {
        "snapshot_date": snapshot_date,
        "model_lineage": run_summary.get("model_lineage"),
        "overall_status": overall_status,
        "warning_count": len(warning_signals),
        "critical_count": len(critical_signals),
        "prediction_psi": prediction_psi,
        "max_feature_psi_including_tenure": max_feature_psi,
        "max_behavioral_feature_psi": max_behavioral_feature_psi,
        "performance_available": performance_summary.get("available"),
        "performance_reference_comparable": performance_summary.get(
            "reference_comparable"
        ),
        "created_at": created_at,
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
    }
    pd.DataFrame([flat_record]).to_csv(
        output_dir / "alert_record.csv", index=False
    )
    pd.DataFrame(signals).to_csv(
        output_dir / "alert_signals.csv", index=False
    )

    print("Azure churn monitoring alert evaluation")
    print("=" * 70)
    print("Snapshot date:", snapshot_date)
    print("Model:", run_summary.get("model_lineage"))
    print("Overall status:", overall_status)
    print("Warnings:", warning_signals)
    print("Critical:", critical_signals)
    print(json.dumps(alert_summary, indent=2, default=str))

    # Intentionally always return normally. Alert state is monitoring output,
    # not a reason to discard successful scoring/retraining artifacts.
    return alert_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-asset-ref", required=True)
    parser.add_argument("--alert-output", required=True)
    parser.add_argument("--performance-data", required=False, default=None)

    parser.add_argument("--prediction-psi-warning", type=float, default=0.10)
    parser.add_argument("--prediction-psi-critical", type=float, default=0.25)
    parser.add_argument("--feature-psi-warning", type=float, default=0.10)
    parser.add_argument("--feature-psi-critical", type=float, default=0.25)

    parser.add_argument("--max-roc-auc-drop", type=float, default=0.005)
    parser.add_argument("--max-recall-drop", type=float, default=0.05)
    parser.add_argument("--max-brier-worsening", type=float, default=0.02)
    parser.add_argument("--critical-multiplier", type=float, default=2.0)

    args = parser.parse_args()
    main(
        score_data=args.score_data,
        model_path=args.model,
        model_asset_ref=args.model_asset_ref,
        alert_output=args.alert_output,
        performance_data=args.performance_data,
        prediction_psi_warning=args.prediction_psi_warning,
        prediction_psi_critical=args.prediction_psi_critical,
        feature_psi_warning=args.feature_psi_warning,
        feature_psi_critical=args.feature_psi_critical,
        max_roc_auc_drop=args.max_roc_auc_drop,
        max_recall_drop=args.max_recall_drop,
        max_brier_worsening=args.max_brier_worsening,
        critical_multiplier=args.critical_multiplier,
    )
