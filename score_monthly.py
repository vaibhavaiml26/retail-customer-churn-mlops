"""Monthly ingestion + official/shadow churn scoring + immediate/delayed monitoring."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import pandas as pd
import config
from drift_monitoring import compute_feature_drift_report, compute_prediction_drift
from feature_engineering import generate_scoring_dataset
from label_resolution import resolve_matured_predictions
from model_registry import load_champion, load_previous_champion
from pipeline_common import (
    MONITORING_HISTORY_PATH,
    alert,
    align_features,
    append_history_row,
    ingest_validate_curate,
    log_run,
    new_run_id,
)


def score_active_customers(
    model,
    threshold,
    X_current,
    customer_ids,
    metadata,
    snapshot_date,
    run_id,
    prediction_role: str = "official",
):
    probs = model.predict_proba(X_current)[:, 1]
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    outcome_end = snapshot_date + pd.DateOffset(months=config.OUTCOME_PERIOD_MONTHS)
    return pd.DataFrame({
        "CustomerID": customer_ids.values,
        "snapshot_date": snapshot_date,
        "outcome_start": snapshot_date,
        "outcome_end": outcome_end,
        "churn_probability": probs,
        "predicted_churn": (probs >= threshold).astype(int),
        "risk_tier": pd.cut(
            probs,
            bins=config.RISK_BINS,
            labels=config.RISK_LABELS,
            include_lowest=True,
        ),
        "model_type": metadata["model_type"],
        "model_version": metadata["version"],
        "prediction_role": prediction_role,
        "threshold_used": threshold,
        "run_id": run_id,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    })


def _write_shadow_predictions(clean_df, X_current, customer_ids, snapshot_date, run_id, current_version):
    """Score the same cohort silently with the immediately previous champion."""
    if not config.SHADOW_SCORING_ENABLED:
        return None, None

    shadow_model, shadow_meta = load_previous_champion(config.MODEL_TYPE)
    if shadow_model is None or shadow_meta.get("version") == current_version:
        return None, None

    try:
        shadow_X = align_features(X_current, shadow_meta["feature_columns"])
        shadow_threshold = float(shadow_meta["metrics"]["threshold"])
        shadow_predictions = score_active_customers(
            shadow_model,
            shadow_threshold,
            shadow_X,
            customer_ids,
            shadow_meta,
            snapshot_date,
            run_id,
            prediction_role="shadow",
        )
        config.SHADOW_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        shadow_path = config.SHADOW_PREDICTIONS_DIR / (
            f"snapshot_{pd.Timestamp(snapshot_date):%Y-%m-%d}__{shadow_meta['version']}.parquet"
        )
        shadow_predictions.to_parquet(shadow_path, index=False)
        return shadow_predictions, shadow_path
    except Exception as exc:
        # Shadow scoring must never block the official production score.
        alert(f"Shadow scoring skipped: {exc}", "warning")
        return None, None


def main(new_file: str, snapshot_date: str | pd.Timestamp):
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    run_id = new_run_id("monthly_score")
    summary = {"stage": "monthly_score", "run_id": run_id, "snapshot_date": str(snapshot_date.date())}

    try:
        clean_df, ingestion, was_new = ingest_validate_curate(new_file, snapshot_date)
        summary["new_ingestion"] = was_new
        summary["source_sha256"] = ingestion.get("sha256")

        champion, metadata = load_champion(config.MODEL_TYPE)
        if champion is None:
            raise RuntimeError("No champion model. Run bootstrap_local.py or retrain_quarterly.py first.")

        X_current, customer_ids = generate_scoring_dataset(clean_df, snapshot_date)
        if X_current.empty:
            raise RuntimeError("No active customers in the scoring feature window.")
        X_current = align_features(X_current, metadata["feature_columns"])
        threshold = float(metadata["metrics"]["threshold"])
        predictions = score_active_customers(
            champion, threshold, X_current, customer_ids, metadata, snapshot_date, run_id,
            prediction_role="official",
        )

        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.PREDICTIONS_DIR / f"snapshot_{snapshot_date:%Y-%m-%d}__{metadata['version']}.parquet"
        predictions.to_parquet(out_path, index=False)
        summary.update({
            "active_customers": len(X_current),
            "model_version": metadata["version"],
            "threshold_used": threshold,
            "predictions_written": str(out_path),
            "predicted_churn_rate": round(float(predictions["predicted_churn"].mean()), 4),
            "mean_churn_probability": round(float(predictions["churn_probability"].mean()), 4),
        })

        shadow_predictions, shadow_path = _write_shadow_predictions(
            clean_df, X_current, customer_ids, snapshot_date, run_id, metadata["version"]
        )
        if shadow_predictions is not None:
            summary.update({
                "shadow_model_version": str(shadow_predictions["model_version"].iloc[0]),
                "shadow_threshold_used": float(shadow_predictions["threshold_used"].iloc[0]),
                "shadow_predictions_written": str(shadow_path),
                "shadow_predicted_churn_rate": round(float(shadow_predictions["predicted_churn"].mean()), 4),
                "shadow_mean_churn_probability": round(float(shadow_predictions["churn_probability"].mean()), 4),
            })

        # Feature drift against the champion's saved training sample. Features
        # such as tenure_days that mechanically accumulate with calendar time
        # are reported separately from behavior-driven drift.
        max_feature_psi = float("nan")
        max_behavioral_feature_psi = float("nan")
        temporal_feature_psi = {}
        sample_path = metadata.get("training_feature_sample_path")
        if sample_path:
            X_base = pd.read_parquet(sample_path)
            X_base = align_features(X_base, metadata["feature_columns"])
            report = compute_feature_drift_report(
                X_base,
                X_current,
                expected_temporal_features=config.EXPECTED_TEMPORAL_DRIFT_FEATURES,
            )
            print(report)
            drift_path = config.MONITORING_DIR / "feature_drift" / f"{snapshot_date:%Y-%m-%d}.csv"
            drift_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(drift_path, index=False)
            max_feature_psi = float(report["psi"].max())

            behavioral = report[report["drift_type"] == "behavioral"]
            if not behavioral.empty:
                max_behavioral_feature_psi = float(behavioral["psi"].max())
                high_behavioral = behavioral[behavioral["psi"] >= config.DRIFT_PSI_HIGH]
                if not high_behavioral.empty:
                    alert(
                        f"High behavioral feature drift: {high_behavioral[['feature','psi']].to_dict('records')}",
                        "warning",
                    )

            temporal = report[report["drift_type"] == "expected_temporal"]
            temporal_feature_psi = {
                str(row.feature): float(row.psi) for row in temporal.itertuples(index=False)
            }

        prediction_psi = float("nan")
        pred_base_path = metadata.get("prediction_reference_sample_path") or metadata.get("training_prediction_sample_path")
        if pred_base_path:
            baseline_probs = pd.read_parquet(pred_base_path)["churn_probability"].to_numpy()
            prediction_psi = compute_prediction_drift(
                baseline_probs, predictions["churn_probability"].to_numpy()
            )
            if prediction_psi >= config.DRIFT_PSI_HIGH:
                alert(f"High prediction drift PSI={prediction_psi:.3f}.", "warning")
            elif prediction_psi >= config.DRIFT_PSI_WARNING:
                alert(f"Moderate prediction drift PSI={prediction_psi:.3f}; monitor trend.", "warning")

        monitoring_row = {
            "run_id": run_id,
            "snapshot_date": str(snapshot_date.date()),
            "model_version": metadata["version"],
            "threshold_used": threshold,
            "active_customers": len(X_current),
            "predicted_churn_rate": float(predictions["predicted_churn"].mean()),
            "mean_churn_probability": float(predictions["churn_probability"].mean()),
            "max_feature_psi": max_feature_psi,
            "max_behavioral_feature_psi": max_behavioral_feature_psi,
            "prediction_psi": prediction_psi,
            **{f"temporal_psi_{k}": v for k, v in temporal_feature_psi.items()},
        }
        if shadow_predictions is not None:
            monitoring_row.update({
                "shadow_model_version": str(shadow_predictions["model_version"].iloc[0]),
                "shadow_predicted_churn_rate": float(shadow_predictions["predicted_churn"].mean()),
                "shadow_mean_churn_probability": float(shadow_predictions["churn_probability"].mean()),
            })
        append_history_row(MONITORING_HISTORY_PATH, monitoring_row)
        summary.update({
            "max_feature_psi": max_feature_psi,
            "max_behavioral_feature_psi": max_behavioral_feature_psi,
            "expected_temporal_feature_psi": temporal_feature_psi,
            "prediction_psi": prediction_psi,
        })

        resolved = resolve_matured_predictions(clean_df, snapshot_date)
        summary["delayed_prediction_sets_evaluated"] = len(resolved)
        summary["delayed_cohorts_evaluated"] = len({r["snapshot_date"] for r in resolved})
        if resolved:
            summary["delayed_metrics"] = resolved

        log_run(summary)
        return summary
    except Exception as exc:
        summary.update({"status": "FAILED", "error": str(exc)})
        log_run(summary)
        alert(str(exc), "error")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-file", required=True, help="Complete monthly transaction CSV")
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Exclusive feature cutoff, normally first day after the batch month, e.g. 2027-10-01",
    )
    args = parser.parse_args()
    main(args.new_file, args.snapshot_date)
