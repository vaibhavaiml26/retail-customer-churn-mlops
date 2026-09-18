"""Backfill a shadow prediction for an already-scored historical cohort.

Useful during the local replay after adding shadow scoring. It recreates the
point-in-time feature set, scores it with the champion that the official model
replaced, stores a shadow prediction file, and resolves it immediately if the
outcome window is already mature.
"""
from __future__ import annotations
import argparse
import pandas as pd

import config
from feature_engineering import generate_scoring_dataset
from ingestion_manifest import latest_data_through_date
from label_resolution import resolve_matured_predictions
from model_registry import get_current_version, get_previous_version, load_model
from pipeline_common import align_features, load_curated, new_run_id
from score_monthly import score_active_customers


def _official_file(snapshot_date: pd.Timestamp):
    matches = sorted(config.PREDICTIONS_DIR.glob(f"snapshot_{snapshot_date:%Y-%m-%d}__*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No official prediction file found for {snapshot_date.date()}.")
    if len(matches) > 1:
        # Prefer the latest file if the same cohort was intentionally rescored.
        return matches[-1]
    return matches[0]


def main(snapshot_date: str | pd.Timestamp):
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    clean_df = load_curated()
    if clean_df.empty:
        raise RuntimeError("Curated data is empty.")

    official_path = _official_file(snapshot_date)
    official = pd.read_parquet(official_path)
    official_version = str(official["model_version"].iloc[0])
    _, official_meta = load_model(config.MODEL_TYPE, official_version)

    shadow_version = official_meta.get("replaced_champion_version")
    if not shadow_version and official_version == get_current_version(config.MODEL_TYPE):
        shadow_version = get_previous_version(config.MODEL_TYPE)
    if not shadow_version:
        raise RuntimeError(
            f"Could not determine the model that preceded official version {official_version}."
        )
    if shadow_version == official_version:
        raise RuntimeError("Resolved shadow version is the same as the official model.")

    shadow_model, shadow_meta = load_model(config.MODEL_TYPE, shadow_version)
    X_current, customer_ids = generate_scoring_dataset(clean_df, snapshot_date)
    X_shadow = align_features(X_current, shadow_meta["feature_columns"])
    threshold = float(shadow_meta["metrics"]["threshold"])
    run_id = new_run_id("shadow_backfill")
    shadow = score_active_customers(
        shadow_model,
        threshold,
        X_shadow,
        customer_ids,
        shadow_meta,
        snapshot_date,
        run_id,
        prediction_role="shadow",
    )

    config.SHADOW_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SHADOW_PREDICTIONS_DIR / f"snapshot_{snapshot_date:%Y-%m-%d}__{shadow_version}.parquet"
    shadow.to_parquet(out_path, index=False)
    print(f"Official model: {official_version}")
    print(f"Shadow model:   {shadow_version}")
    print(f"Shadow predictions written: {out_path}")

    coverage = latest_data_through_date()
    outcome_end = snapshot_date + pd.DateOffset(months=config.OUTCOME_PERIOD_MONTHS)
    if coverage is not None and coverage >= outcome_end:
        resolved = resolve_matured_predictions(clean_df, coverage)
        newly_resolved = [
            row for row in resolved
            if row.get("snapshot_date") == str(snapshot_date.date())
            and row.get("prediction_role") == "shadow"
        ]
        if newly_resolved:
            print("Shadow cohort resolved immediately:")
            print(pd.DataFrame(newly_resolved).to_string(index=False))
        else:
            print("Shadow cohort was already evaluated or no new evaluation was required.")
    else:
        print(f"Outcome is not mature yet. Need data through {outcome_end.date()}.")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, help="Existing official cohort, e.g. 2011-09-01")
    args = parser.parse_args()
    main(args.snapshot_date)
