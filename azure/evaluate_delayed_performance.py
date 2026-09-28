"""Azure ML delayed ground-truth evaluation for monthly retail-churn predictions.

For each monthly evaluation run:

    evaluation_snapshot = as_of_date - outcome_months

Example for this project:
    as_of_date          = 2011-09-01
    outcome_months      = 3
    evaluation_snapshot = 2011-06-01
    outcome window      = [2011-06-01, 2011-09-01)

The script loads the prediction file generated for the evaluation snapshot and
resolves the true `inactive_90d` target from future transactions.

IMPORTANT LABEL RULE
--------------------
A customer is ACTIVE (inactive_90d = 0) only if at least one transaction with
Quantity > 0 exists in the forward outcome window.

Returns/cancellations (Quantity < 0) by themselves DO NOT count as activity.

Expected scoring prediction columns from score_azure.py:
    CustomerID
    snapshot_date
    churn_probability
    predicted_churn
    model_version
    threshold_used
    scored_at
    azure_run_id

Outputs:
    performance_metrics_<snapshot>.json
    performance_record_<snapshot>.csv
    evaluated_predictions_<snapshot>.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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

# This file is expected under project/azure/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import clean_transactions, load_transactions
from data_validation import validate_schema


REQUIRED_PREDICTION_COLUMNS = {
    "CustomerID",
    "snapshot_date",
    "churn_probability",
    "predicted_churn",
}


def _parse_date(value: str, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid {label}: {value!r}")
    return ts.normalize()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def _resolve_csv_files(input_path: str, label: str) -> list[str]:
    """Resolve an Azure ML uri_folder (or local CSV) into CSV files."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"{label} input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{label} input is not a CSV file: {path}")
        files = [str(path)]
    else:
        files = sorted(
            str(file)
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() == ".csv"
        )

    if not files:
        raise ValueError(f"No CSV files found under {label} input: {path}")

    print(f"{label}: found {len(files)} CSV file(s)")
    for file in files:
        print(f"  {file}")

    return files


def _resolve_prediction_file(
    predictions_data: str,
    evaluation_snapshot: pd.Timestamp,
) -> Path | None:
    def _resolve_prediction_file(
    predictions_data: str,
    evaluation_snapshot: pd.Timestamp,
) -> Path | None:
    """
    Find the latest predictions_YYYY-MM-DD.parquet file for the
    requested prediction snapshot.

    Multiple files may exist because the same snapshot can be rerun.
    Prediction history is expected to use execution folders whose names
    begin with a UTC timestamp, for example:

        prediction-history/
        └── 2011-06-01/
            ├── 20260927T120000Z-local-a1b2c3d4/
            │   └── predictions_2011-06-01.parquet
            └── 20260928T090000Z-github-123456789-1/
                └── predictions_2011-06-01.parquet

    The latest execution folder is selected.

    Return None when no persisted prediction exists for the requested
    snapshot so delayed evaluation can skip gracefully.
    """

    snapshot_str = pd.Timestamp(
        evaluation_snapshot
    ).strftime("%Y-%m-%d")

    prediction_files = list(
        Path(predictions_data).rglob(
            f"predictions_{snapshot_str}.parquet"
        )
    )

    if not prediction_files:
        return None

    prediction_file = sorted(
        prediction_files,
        key=lambda p: p.parent.name,
    )[-1]

    return prediction_file



    return prediction_file


def _single_value_or_none(series: pd.Series):
    values = series.dropna().unique().tolist()
    if len(values) == 1:
        value = values[0]
        return value.item() if hasattr(value, "item") else value
    return None


def _safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _safe_pr_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(average_precision_score(y_true, probabilities))


def _write_skip_outputs(
    output_dir: Path,
    as_of_date: pd.Timestamp,
    evaluation_snapshot: pd.Timestamp,
    first_prediction_snapshot: pd.Timestamp,
    reason: str,
) -> dict:
    """Create a valid output when no prediction snapshot is mature yet."""
    payload = {
        "stage": "azure_delayed_performance",
        "evaluation_skipped": True,
        "reason": reason,
        "as_of_date": as_of_date.date().isoformat(),
        "evaluation_snapshot": evaluation_snapshot.date().isoformat(),
        "first_prediction_snapshot": first_prediction_snapshot.date().isoformat(),
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = (
        output_dir
        / f"performance_metrics_{evaluation_snapshot:%Y-%m-%d}.json"
    )
    _write_json(metrics_path, payload)

    pd.DataFrame([payload]).to_csv(
        output_dir
        / f"performance_record_{evaluation_snapshot:%Y-%m-%d}.csv",
        index=False,
    )

    print("\nDelayed performance evaluation skipped.")
    print(json.dumps(payload, indent=2))
    return payload


def main(
    bootstrap_data: str,
    replay_data: str,
    predictions_data: str,
    performance_output: str,
    as_of_date: str,
    outcome_months: int = 3,
    first_prediction_snapshot: str = "2011-06-01",
) -> dict:
    as_of_ts = _parse_date(as_of_date, "as-of date")
    first_prediction_ts = _parse_date(
        first_prediction_snapshot,
        "first prediction snapshot",
    )

    if outcome_months <= 0:
        raise ValueError("outcome_months must be greater than zero.")

    evaluation_snapshot = (
        as_of_ts - pd.DateOffset(months=outcome_months)
    ).normalize()

    output_dir = Path(performance_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Azure delayed churn performance evaluation")
    print("=" * 70)
    print(f"As-of date:                 {as_of_ts.date()}")
    print(f"Outcome months:             {outcome_months}")
    print(f"Prediction snapshot:        {evaluation_snapshot.date()}")
    print(
        "Outcome window:             "
        f"[{evaluation_snapshot.date()} -> {as_of_ts.date()})"
    )
    print(f"First prediction snapshot:  {first_prediction_ts.date()}")
    print(f"Performance output:         {output_dir}")

    # ------------------------------------------------------------------
    # 0. Before the first prediction can possibly mature, exit cleanly.
    # ------------------------------------------------------------------
    if evaluation_snapshot < first_prediction_ts:
        reason = (
            "No prediction snapshot is mature yet: "
            f"evaluation_snapshot={evaluation_snapshot.date()} is earlier than "
            f"first_prediction_snapshot={first_prediction_ts.date()}."
        )
        return _write_skip_outputs(
            output_dir=output_dir,
            as_of_date=as_of_ts,
            evaluation_snapshot=evaluation_snapshot,
            first_prediction_snapshot=first_prediction_ts,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 1. Load the historical prediction snapshot from exactly N months ago.
    # ------------------------------------------------------------------
    prediction_file = _resolve_prediction_file(
        predictions_data,
        evaluation_snapshot,
    )
    if prediction_file is None:
        reason = (
            "No persisted prediction snapshot was found for the matured "
            f"evaluation snapshot {evaluation_snapshot.date()} under "
            f"predictions_data={predictions_data!r}. Expected file name: "
            f"predictions_{evaluation_snapshot.strftime('%Y-%m-%d')}.parquet."
        )
        return _write_skip_outputs(
            output_dir=output_dir,
            as_of_date=as_of_ts,
            evaluation_snapshot=evaluation_snapshot,
            first_prediction_snapshot=first_prediction_ts,
            reason=reason,
        )

    predictions = pd.read_parquet(prediction_file)
    prediction_execution_id = prediction_file.parent.name

    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {missing}. "
            f"Columns found: {list(predictions.columns)}"
        )

    predictions["snapshot_date"] = pd.to_datetime(
        predictions["snapshot_date"]
    ).dt.normalize()

    predictions = predictions[
        predictions["snapshot_date"] == evaluation_snapshot
    ].copy()

    if predictions.empty:
        raise ValueError(
            f"Prediction file {prediction_file} contains no rows for "
            f"snapshot {evaluation_snapshot.date()}."
        )

    if predictions["CustomerID"].isna().any():
        raise ValueError("Prediction file contains missing CustomerID values.")

    if predictions["CustomerID"].duplicated().any():
        duplicates = int(predictions["CustomerID"].duplicated().sum())
        raise ValueError(
            f"Prediction file contains {duplicates} duplicate CustomerID row(s). "
            "Expected one prediction per customer per snapshot."
        )

    predictions["churn_probability"] = pd.to_numeric(
        predictions["churn_probability"],
        errors="raise",
    )
    predictions["predicted_churn"] = pd.to_numeric(
        predictions["predicted_churn"],
        errors="raise",
    ).astype(int)

    invalid_probs = ~predictions["churn_probability"].between(0.0, 1.0)
    if invalid_probs.any():
        raise ValueError(
            f"Prediction file contains {int(invalid_probs.sum())} probability "
            "value(s) outside [0, 1]."
        )

    invalid_classes = ~predictions["predicted_churn"].isin([0, 1])
    if invalid_classes.any():
        raise ValueError(
            f"Prediction file contains {int(invalid_classes.sum())} "
            "predicted_churn value(s) outside {0, 1}."
        )

    print("\nPrediction snapshot")
    print("-" * 70)
    print(f"Prediction file:      {prediction_file}")
    print(f"Customers predicted:  {len(predictions):,}")

    # ------------------------------------------------------------------
    # 2. Reconstruct ONLY transaction history available by the evaluation
    #    date. Later replay months may exist in the data asset, but they must
    #    not leak into this historical evaluation.
    # ------------------------------------------------------------------
    bootstrap_files = _resolve_csv_files(bootstrap_data, "Bootstrap")
    replay_files = _resolve_csv_files(replay_data, "Replay")

    bootstrap_df = load_transactions(bootstrap_files)
    replay_df = load_transactions(replay_files)

    bootstrap_to_date = bootstrap_df[
        bootstrap_df["InvoiceDate"] < as_of_ts
    ].copy()
    replay_to_date = replay_df[
        replay_df["InvoiceDate"] < as_of_ts
    ].copy()

    if not bootstrap_to_date.empty and not replay_to_date.empty:
        bootstrap_max = bootstrap_to_date["InvoiceDate"].max()
        replay_min = replay_to_date["InvoiceDate"].min()

        if bootstrap_max >= replay_min:
            raise ValueError(
                "Bootstrap and replay inputs overlap in time: "
                f"bootstrap max={bootstrap_max}, replay min={replay_min}. "
                "Refusing to double-count historical transactions."
            )

    raw_df = (
        pd.concat(
            [bootstrap_to_date, replay_to_date],
            ignore_index=True,
        )
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )

    if raw_df.empty:
        raise ValueError(
            f"No transactions are available before {as_of_ts.date()}."
        )

    validation = validate_schema(raw_df)
    print("\nEvaluation-history validation")
    print("-" * 70)
    print(validation.report())

    if not validation.passed:
        raise ValueError(
            "Delayed-performance transaction history failed validation."
        )

    clean_df = clean_transactions(raw_df)

    print(f"Raw rows available:   {len(raw_df):,}")
    print(f"Clean rows available: {len(clean_df):,}")
    print(
        "History date range:  "
        f"{clean_df['InvoiceDate'].min()} -> "
        f"{clean_df['InvoiceDate'].max()}"
    )

    # ------------------------------------------------------------------
    # 3. Resolve ground truth.
    #
    #    inactive_90d = 0:
    #        customer has at least one Quantity > 0 sale during the outcome
    #        window.
    #
    #    inactive_90d = 1:
    #        customer has NO positive-quantity sale during the outcome window.
    #
    #    Return-only activity therefore does NOT make a customer active.
    # ------------------------------------------------------------------
    outcome_df = clean_df[
        (clean_df["InvoiceDate"] >= evaluation_snapshot)
        & (clean_df["InvoiceDate"] < as_of_ts)
    ].copy()

    positive_sales = outcome_df[outcome_df["Quantity"] > 0]

    active_customer_ids = set(
        positive_sales["CustomerID"].dropna().unique().tolist()
    )

    evaluated = predictions.copy()
    evaluated["actual_inactive_90d"] = (
        ~evaluated["CustomerID"].isin(active_customer_ids)
    ).astype(int)

    y_true = evaluated["actual_inactive_90d"]
    probabilities = evaluated["churn_probability"]
    predicted = evaluated["predicted_churn"]

    # ------------------------------------------------------------------
    # 4. Calculate delayed production-performance metrics.
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    roc_auc = _safe_roc_auc(y_true, probabilities)
    pr_auc = _safe_pr_auc(y_true, probabilities)

    threshold = (
        _single_value_or_none(evaluated["threshold_used"])
        if "threshold_used" in evaluated.columns
        else None
    )
    model_version = (
        _single_value_or_none(evaluated["model_version"])
        if "model_version" in evaluated.columns
        else None
    )
    scoring_run_id = (
        _single_value_or_none(evaluated["azure_run_id"])
        if "azure_run_id" in evaluated.columns
        else None
    )

    evaluated_at = datetime.now(timezone.utc).isoformat()

    metrics = {
        "stage": "azure_delayed_performance",
        "evaluation_skipped": False,
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "scoring_run_id": scoring_run_id,
        "prediction_execution_id": prediction_execution_id,
        "as_of_date": as_of_ts.date().isoformat(),
        "prediction_snapshot": evaluation_snapshot.date().isoformat(),
        "outcome_start": evaluation_snapshot.date().isoformat(),
        "outcome_end_exclusive": as_of_ts.date().isoformat(),
        "outcome_months": int(outcome_months),
        "model_lineage": model_version,
        "threshold": float(threshold) if threshold is not None else None,
        "customers_evaluated": int(len(evaluated)),
        "customers_with_positive_sale_in_outcome": int(
            evaluated["CustomerID"].isin(active_customer_ids).sum()
        ),
        "actual_churn_rate": float(y_true.mean()),
        "predicted_churn_rate": float(predicted.mean()),
        "mean_churn_probability": float(probabilities.mean()),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "prediction_file": prediction_file.name,
        "evaluated_at": evaluated_at,
    }

    print("\nDelayed performance metrics")
    print("-" * 70)
    for key in [
        "customers_evaluated",
        "actual_churn_rate",
        "predicted_churn_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("confusion_matrix:")
    print(cm)

    # ------------------------------------------------------------------
    # 5. Persist both detailed and one-row monitoring artifacts.
    # ------------------------------------------------------------------
    snapshot_label = evaluation_snapshot.strftime("%Y-%m-%d")

    evaluated.to_parquet(
        output_dir / f"evaluated_predictions_{snapshot_label}.parquet",
        index=False,
    )

    _write_json(
        output_dir / f"performance_metrics_{snapshot_label}.json",
        metrics,
    )

    performance_record = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

    pd.DataFrame([performance_record]).to_csv(
        output_dir / f"performance_record_{snapshot_label}.csv",
        index=False,
    )

    print("\nAzure delayed performance evaluation completed successfully.")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bootstrap-data", required=True)
    parser.add_argument("--replay-data", required=True)

    parser.add_argument(
        "--predictions-data",
        required=True,
        help=(
            "Folder containing historical predictions_YYYY-MM-DD.parquet "
            "files, or a scoring job's predictions_output folder."
        ),
    )

    parser.add_argument(
        "--performance-output",
        required=True,
        help="Azure ML output folder for delayed performance artifacts.",
    )

    parser.add_argument(
        "--as-of-date",
        required=True,
        help=(
            "Date when ground truth is available. Example: 2011-09-01 "
            "evaluates predictions from 2011-06-01 when outcome_months=3."
        ),
    )

    parser.add_argument(
        "--outcome-months",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--first-prediction-snapshot",
        default="2011-06-01",
        help=(
            "Earliest monthly prediction snapshot. Runs before this snapshot "
            "has matured exit successfully with evaluation_skipped=true."
        ),
    )

    args = parser.parse_args()

    main(
        bootstrap_data=args.bootstrap_data,
        replay_data=args.replay_data,
        predictions_data=args.predictions_data,
        performance_output=args.performance_output,
        as_of_date=args.as_of_date,
        outcome_months=args.outcome_months,
        first_prediction_snapshot=args.first_prediction_snapshot,
    )
