import sys
from pathlib import Path

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Azure ML bootstrap training entry point for the retail churn project.

This reuses the project's existing cleaning, feature engineering, snapshot,
model, threshold and evaluation logic. Azure supplies the historical bootstrap
CSV as an input and a writable model-output folder as an output.
"""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost



import config
from data_loader import clean_transactions, load_transactions
from data_validation import validate_schema
from dataset_builder import build_rolling_train_val_test
from feature_engineering import max_available_snapshots
from models import (
    build_production_candidate,
    evaluate_fitted_model,
    find_best_threshold,
    get_feature_importance,
    temporal_cv_roc_auc,
)


from pathlib import Path


def _resolve_bootstrap_files(input_path: str) -> list[str]:
    """
    Resolve an Azure ML uri_folder into a list of CSV file paths.
    """

    print("Azure bootstrap path received:", input_path)

    if not input_path:
        raise ValueError(
            "bootstrap_data argument is empty or None."
        )

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Bootstrap input path does not exist: {path}"
        )

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Bootstrap input file is not a CSV: {path}"
            )

        files = [str(path)]

    else:
        files = sorted(
            str(file)
            for file in path.rglob("*")
            if file.is_file()
            and file.suffix.lower() == ".csv"
        )

    if not files:
        raise ValueError(
            f"No CSV files found in bootstrap input: {path}"
        )

    print(f"Found {len(files)} CSV file(s):")

    for file in files:
        print(f"  {file}")

    print("Resolver returning:", files)
    print("Resolver return type:", type(files))

    # THIS MUST BE OUTSIDE ALL IF/ELSE BLOCKS
    return files

def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main(bootstrap_data: str, model_output: str) -> dict:
    print("Bootstrap data argument:", bootstrap_data)
    print("Model output:", model_output)

    input_files = _resolve_bootstrap_files(bootstrap_data)
    output_dir = Path(model_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bootstrap input: {input_files}")
    print(f"Model output: {output_dir}")

    if input_files is None:
        raise RuntimeError(
            "_resolve_bootstrap_files returned None"
        )

    raw_df = load_transactions(input_files)
    print("\nBootstrap input summary")
    print("-----------------------")
    print(f"Files loaded: {len(input_files)}")
    print(f"Raw rows: {len(raw_df):,}")
    print(
        "Date range:",
        raw_df["InvoiceDate"].min(),
        "to",
        raw_df["InvoiceDate"].max(),
    )

    validation = validate_schema(raw_df)
    print(validation.report())
    if not validation.passed:
        raise ValueError(validation.report())

    clean_df = clean_transactions(raw_df)
    print(f"Clean rows: {len(clean_df):,}")
    available = max_available_snapshots(clean_df, config.SNAPSHOT_STEP_MONTHS)
    needed = (
        config.N_TRAIN_SNAPSHOTS
        + config.N_VAL_SNAPSHOTS
        + config.N_TEST_SNAPSHOTS
    )
    print(f"Fully labeled snapshots available: {available}; needed: {needed}")
    if available < needed:
        raise ValueError(
            f"Bootstrap asset supports only {available} fully labeled snapshots; "
            f"the configured {config.N_TRAIN_SNAPSHOTS}/{config.N_VAL_SNAPSHOTS}/"
            f"{config.N_TEST_SNAPSHOTS} split needs {needed}."
        )

    bundle = build_rolling_train_val_test(
        clean_df,
        config.N_TRAIN_SNAPSHOTS,
        config.N_VAL_SNAPSHOTS,
        config.N_TEST_SNAPSHOTS,
        step_months=config.SNAPSHOT_STEP_MONTHS,
        end_snapshot=available,
    )

    model, params = build_production_candidate(config.MODEL_TYPE)
    cv_auc = temporal_cv_roc_auc(
        model,
        bundle.train.X,
        bundle.train.y,
        bundle.train.snapshot_ids,
    )
    print(f"Temporal CV ROC-AUC: {cv_auc:.6f}")

    model.fit(bundle.train.X, bundle.train.y)

    threshold, threshold_table = find_best_threshold(
        model,
        bundle.val.X,
        bundle.val.y,
        metric=config.THRESHOLD_METRIC,
    )
    val_metrics = evaluate_fitted_model(
        model,
        bundle.val.X,
        bundle.val.y,
        label="Azure bootstrap validation",
        threshold=threshold,
    )
    val_metrics["cv_roc_auc"] = float(cv_auc)

    test_metrics = evaluate_fitted_model(
        model,
        bundle.test.X,
        bundle.test.y,
        label="Azure bootstrap held-out test",
        threshold=threshold,
    )

    # Save the fitted model and the same baseline artifacts used by monthly
    # drift monitoring in the local implementation.
    joblib.dump(model, output_dir / "model.joblib")

    sample_n = min(config.DRIFT_BASELINE_SAMPLE_SIZE, len(bundle.train.X))
    training_sample = bundle.train.X.sample(
        sample_n, random_state=config.RANDOM_STATE
    )
    training_sample.to_parquet(
        output_dir / "training_feature_sample.parquet", index=False
    )
    pd.DataFrame(
        {"churn_probability": model.predict_proba(training_sample)[:, 1]}
    ).to_parquet(output_dir / "training_prediction_sample.parquet", index=False)

    pd.DataFrame(
        {"churn_probability": model.predict_proba(bundle.val.X)[:, 1]}
    ).to_parquet(output_dir / "prediction_reference_sample.parquet", index=False)

    threshold_table.to_csv(output_dir / "threshold_sweep.csv", index=False)
    get_feature_importance(model, bundle.train.X.columns).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_run_id": os.getenv("AZUREML_RUN_ID"),
        "model_type": config.MODEL_TYPE,
        "params": params,
        "threshold": float(threshold),
        "threshold_metric": config.THRESHOLD_METRIC,
        "metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_columns": list(bundle.train.X.columns),
        "training_snapshots": bundle.train_snapshots,
        "validation_snapshots": bundle.val_snapshots,
        "test_snapshots": bundle.test_snapshots,
        "latest_snapshot_used": bundle.latest_snapshot,
        "snapshot_step_months": config.SNAPSHOT_STEP_MONTHS,
        "feature_period_months": config.FEATURE_PERIOD_MONTHS,
        "outcome_period_months": config.OUTCOME_PERIOD_MONTHS,
        "bootstrap_rows_raw": int(len(raw_df)),
        "bootstrap_rows_clean": int(len(clean_df)),
        "bootstrap_date_min": str(clean_df["InvoiceDate"].min()),
        "bootstrap_date_max": str(clean_df["InvoiceDate"].max()),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    _json_dump(output_dir / "metadata.json", metadata)

    print("\nAzure bootstrap completed successfully.")
    print(json.dumps(metadata, indent=2, default=str))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-data", required=True)
    parser.add_argument("--model-output", required=True)
    args = parser.parse_args()
    main(args.bootstrap_data, args.model_output)
