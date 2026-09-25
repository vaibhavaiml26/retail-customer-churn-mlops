"""Azure ML monthly batch scoring entry point for the retail churn project.

The job is intentionally stateless. For a requested snapshot date it rebuilds
all transaction history that would have been available by that date from:

1. the bootstrap historical data folder, and
2. the replay-months folder.

It then generates the existing production scoring features, loads the
registered Azure ML custom-model folder, scores active customers, and writes
predictions plus monitoring artifacts to the Azure ML output folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# bootstrap_azure.py and score_azure.py live under project/azure/.  Add the
# project root so the existing project modules remain importable when Azure
# runs this file as `python azure/score_azure.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from data_loader import clean_transactions, load_transactions
from data_validation import validate_schema
from drift_monitoring import compute_feature_drift_report, compute_prediction_drift
from feature_engineering import generate_scoring_dataset


def _resolve_csv_files(input_path: str, label: str) -> list[str]:
    """Resolve an Azure ML uri_folder (or a local file for testing) to CSV paths."""
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


def _resolve_model_artifacts(model_path: str) -> tuple[Path, Path, Path]:
    """Find model.joblib, metadata.json, and the registered-model root folder."""
    root = Path(model_path)
    if not root.exists():
        raise FileNotFoundError(f"Registered model input does not exist: {root}")

    model_candidates = list(root.rglob("model.joblib"))
    metadata_candidates = list(root.rglob("metadata.json"))

    if len(model_candidates) != 1:
        raise ValueError(
            f"Expected exactly one model.joblib under registered model input, "
            f"found {len(model_candidates)}: {model_candidates}"
        )
    if len(metadata_candidates) != 1:
        raise ValueError(
            f"Expected exactly one metadata.json under registered model input, "
            f"found {len(metadata_candidates)}: {metadata_candidates}"
        )

    return model_candidates[0], metadata_candidates[0], root


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parse_snapshot_date(value: str) -> pd.Timestamp:
    snapshot_date = pd.Timestamp(value)
    if pd.isna(snapshot_date):
        raise ValueError(f"Invalid snapshot date: {value!r}")
    return snapshot_date.normalize()


def _model_threshold(metadata: dict) -> float:
    """Read the threshold stored by bootstrap_azure.py, with compatibility fallbacks."""
    if "threshold" in metadata:
        return float(metadata["threshold"])

    metrics = metadata.get("metrics", {})
    if "threshold" in metrics:
        return float(metrics["threshold"])

    raise KeyError(
        "Registered model metadata does not contain a classification threshold. "
        "Expected top-level 'threshold' (current Azure bootstrap format) or "
        "metrics['threshold']."
    )


def _model_version_label(metadata: dict, model_asset_ref: str) -> str:
    """Return the registered Azure ML model asset reference used for scoring."""
    if model_asset_ref:
        return model_asset_ref
    if metadata.get("version"):
        return str(metadata["version"])
    if metadata.get("azure_run_id"):
        return str(metadata["azure_run_id"])
    return "unknown"

'''def resolve_as_of_date(value: str) -> pd.Timestamp:
    if value.strip().upper() == "AUTO":
        today = datetime.now(timezone.utc).date()
        first_day = today.replace(day=1)
        return pd.Timestamp(first_day)

    return pd.Timestamp(value)
'''
def main(
    bootstrap_data: str,
    replay_data: str,
    model_path: str,
    predictions_output: str,
    snapshot_date: str,
    model_asset_ref: str,
) -> dict:
    snapshot_ts = _parse_snapshot_date(snapshot_date)
    output_dir = Path(predictions_output)
    output_dir.mkdir(parents=True, exist_ok=True)


    print("Azure monthly churn scoring")
    print("=" * 70)
    print(f"Snapshot date: {snapshot_ts.date()}")
    print(f"Bootstrap input: {bootstrap_data}")
    print(f"Replay input: {replay_data}")
    print(f"Registered model input: {model_path}")
    print(f"Predictions output: {output_dir}")

    # ------------------------------------------------------------------
    # 1. Reconstruct only the history available at the requested cutoff.
    # ------------------------------------------------------------------
    bootstrap_files = _resolve_csv_files(bootstrap_data, "Bootstrap")
    replay_files = _resolve_csv_files(replay_data, "Replay")

    bootstrap_df = load_transactions(bootstrap_files)
    replay_df = load_transactions(replay_files)

    # Critical anti-leakage rule: even when the replay data asset contains
    # later months, this run may only see transactions strictly before the
    # scoring snapshot date.
    bootstrap_to_date = bootstrap_df[bootstrap_df["InvoiceDate"] < snapshot_ts].copy()
    replay_to_date = replay_df[replay_df["InvoiceDate"] < snapshot_ts].copy()

    # Bootstrap and replay are intentionally non-overlapping sources. If their
    # available date ranges overlap, fail rather than silently double-counting
    # transactions in a historical replay.
    if not bootstrap_to_date.empty and not replay_to_date.empty:
        bootstrap_max = bootstrap_to_date["InvoiceDate"].max()
        replay_min = replay_to_date["InvoiceDate"].min()
        if bootstrap_max >= replay_min:
            raise ValueError(
                "Bootstrap and replay inputs overlap in time: "
                f"bootstrap max={bootstrap_max}, replay min={replay_min}. "
                "Bootstrap should end before the replay period begins."
            )

    raw_df = (
        pd.concat([bootstrap_to_date, replay_to_date], ignore_index=True)
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )

    if raw_df.empty:
        raise ValueError(f"No transactions are available before {snapshot_ts.date()}")

    print("\nScoring-history summary")
    print("-" * 70)
    print(f"Bootstrap rows available: {len(bootstrap_to_date):,}")
    print(f"Replay rows available:    {len(replay_to_date):,}")
    print(f"Combined raw rows:        {len(raw_df):,}")
    print(
        f"Combined date range:      {raw_df['InvoiceDate'].min()} -> "
        f"{raw_df['InvoiceDate'].max()}"
    )

    validation = validate_schema(raw_df)
    print(validation.report())
    if not validation.passed:
        raise ValueError(validation.report())

    clean_df = clean_transactions(raw_df)
    print(f"Clean rows:               {len(clean_df):,}")

    # ------------------------------------------------------------------
    # 2. Generate the exact same feature set used by local scoring.
    # ------------------------------------------------------------------
    X_current, customer_ids = generate_scoring_dataset(clean_df, snapshot_ts)

    if X_current.empty:
        raise ValueError(
            f"No active customers were produced for scoring as of {snapshot_ts.date()}"
        )

    # ------------------------------------------------------------------
    # 3. Load the registered custom-model folder and enforce schema parity.
    # ------------------------------------------------------------------
    model_file, metadata_file, model_root = _resolve_model_artifacts(model_path)
    model = joblib.load(model_file)
    metadata = _load_json(metadata_file)

    expected_columns = metadata.get("feature_columns")
    if expected_columns:
        missing = [col for col in expected_columns if col not in X_current.columns]
        unexpected = [col for col in X_current.columns if col not in expected_columns]
        if missing or unexpected:
            raise ValueError(
                "Scoring/training feature mismatch. "
                f"Missing={missing}; unexpected={unexpected}"
            )
        # Enforce training order as well as names. XGBoost is not the place
        # to discover that a DataFrame was shuffled by an innocent refactor.
        X_current = X_current[expected_columns]

    threshold = _model_threshold(metadata)
    model_version = _model_version_label(metadata, model_asset_ref)

    print("\nModel summary")
    print("-" * 70)
    print(f"Model file:      {model_file}")
    print(f"Metadata file:   {metadata_file}")
    print(f"Model type:      {metadata.get('model_type', 'unknown')}")
    print(f"Model lineage:   {model_version}")
    print(f"Threshold:       {threshold:.4f}")
    print(f"Customers scored:{len(X_current):,}")

    # ------------------------------------------------------------------
    # 4. Score customers and persist predictions.
    # ------------------------------------------------------------------
    probabilities = model.predict_proba(X_current)[:, 1]
    predicted_churn = (probabilities >= threshold).astype(int)
    scored_at = datetime.now(timezone.utc).isoformat()

    predictions = pd.DataFrame(
        {
            "CustomerID": customer_ids.values,
            "snapshot_date": snapshot_ts.date().isoformat(),
            "churn_probability": probabilities,
            "predicted_churn": predicted_churn,
            "risk_tier": pd.cut(
                probabilities,
                bins=[0.0, 0.3, 0.6, 1.0],
                labels=["low", "medium", "high"],
                include_lowest=True,
            ),
            "model_version": model_version,
            "threshold_used": threshold,
            "scored_at": scored_at,
            "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        }
    )

    prediction_filename = f"predictions_{snapshot_ts:%Y-%m-%d}.parquet"
    prediction_path = output_dir / prediction_filename
    predictions.to_parquet(prediction_path, index=False)

    # Retaining the scoring feature snapshot is useful for auditing and later
    # delayed evaluation. It also makes debugging feature drift far less mystical.
    feature_snapshot = X_current.copy()
    feature_snapshot.insert(0, "CustomerID", customer_ids.values)
    feature_snapshot.insert(1, "snapshot_date", snapshot_ts.date().isoformat())
    feature_snapshot.to_parquet(
        output_dir / f"features_{snapshot_ts:%Y-%m-%d}.parquet", index=False
    )

    # ------------------------------------------------------------------
    # 5. Immediate monitoring against artifacts bundled with the model.
    # ------------------------------------------------------------------
    max_feature_psi = None
    prediction_psi = None
    max_behavioral_feature_psi = None

    training_sample_candidates = list(model_root.rglob("training_feature_sample.parquet"))
    if len(training_sample_candidates) == 1:
        X_baseline = pd.read_parquet(training_sample_candidates[0])
        common_columns = [c for c in X_baseline.columns if c in X_current.columns]
        feature_drift = compute_feature_drift_report(
            X_baseline[common_columns], X_current[common_columns]
        )
        if not feature_drift.empty:
            max_feature_psi = float(feature_drift["psi"].max())

            behavioral_drift = feature_drift[
                feature_drift["feature"] != "tenure_days"
                ]

            if not behavioral_drift.empty:
                max_behavioral_feature_psi = float(
                    behavioral_drift["psi"].max()
                )

        feature_drift.to_csv(output_dir / "feature_drift.csv", index=False)
        if not feature_drift.empty:
            max_feature_psi = float(feature_drift["psi"].max())
            print(f"Max feature PSI: {max_feature_psi:.6f}")

    prediction_reference_candidates = list(
        model_root.rglob("prediction_reference_sample.parquet")
    )
    if len(prediction_reference_candidates) == 1:
        prediction_reference = pd.read_parquet(prediction_reference_candidates[0])
        if "churn_probability" in prediction_reference.columns:
            prediction_psi = compute_prediction_drift(
                prediction_reference["churn_probability"].to_numpy(),
                probabilities,
            )
            print(f"Prediction PSI: {prediction_psi:.6f}")

    summary = {
        "stage": "azure_monthly_score",
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "snapshot_date": snapshot_ts.date().isoformat(),
        "history_date_min": str(clean_df["InvoiceDate"].min()),
        "history_date_max": str(clean_df["InvoiceDate"].max()),
        "bootstrap_files": len(bootstrap_files),
        "replay_files": len(replay_files),
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "active_customers": int(len(X_current)),
        "model_type": metadata.get("model_type"),
        "model_lineage": model_version,
        "training_run_id": metadata.get("azure_run_id"),
        "threshold": float(threshold),
        "mean_churn_probability": float(probabilities.mean()),
        "predicted_churn_rate": float(predicted_churn.mean()),
        "max_feature_psi": max_feature_psi,
        "max_behavioral_feature_psi": max_behavioral_feature_psi,
        "prediction_psi": prediction_psi,
        "prediction_file": prediction_filename,
        "scored_at": scored_at,
    }
    _write_json(output_dir / "run_summary.json", summary)

    print("\nAzure monthly scoring completed successfully.")
    print(json.dumps(summary, indent=2, default=str))
    monitoring_record = pd.DataFrame([{
        "snapshot_date": snapshot_ts.date().isoformat(),
        "model_lineage": model_version,
        "threshold": float(threshold),
        "active_customers": int(len(X_current)),
        "mean_churn_probability": float(probabilities.mean()),
        "predicted_churn_rate": float(predicted_churn.mean()),
        "max_feature_psi": max_feature_psi,
        "max_behavioral_feature_psi": max_behavioral_feature_psi,
        "prediction_psi": prediction_psi,
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "scored_at": scored_at,
    }])

    monitoring_record.to_csv(
        output_dir / "monitoring_record.csv",
        index=False,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-data", required=True)
    parser.add_argument("--replay-data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-asset-ref", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Scoring cutoff, e.g. 2011-07-01 after June data has arrived.",
    )
    args = parser.parse_args()

    main(
        bootstrap_data=args.bootstrap_data,
        replay_data=args.replay_data,
        model_path=args.model,
        predictions_output=args.predictions_output,
        snapshot_date=args.snapshot_date,
        model_asset_ref=args.model_asset_ref,
    )
