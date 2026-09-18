"""Train a rolling quarterly challenger only when a new labeled snapshot exists."""
from __future__ import annotations
import pandas as pd
import config
from dataset_builder import build_rolling_train_val_test
from drift_monitoring import should_promote_challenger
from feature_engineering import max_available_snapshots
from ingestion_manifest import latest_data_through_date
from model_registry import load_champion, save_model, set_current_version
from models import (
    build_production_candidate,
    evaluate_fitted_model,
    find_best_threshold,
    temporal_cv_roc_auc,
    tune_random_forest,
    tune_xgboost,
)
from pipeline_common import align_features, alert, load_curated, log_run, new_run_id

TUNERS = {"random_forest": tune_random_forest, "xgboost": tune_xgboost}


def latest_ready_snapshot(clean_df: pd.DataFrame) -> tuple[int, pd.Timestamp | None]:
    coverage = latest_data_through_date()
    available = max_available_snapshots(
        clean_df,
        config.SNAPSHOT_STEP_MONTHS,
        data_through_date=coverage,
    )
    return available, coverage


def main(force: bool = False):
    run_id = new_run_id("quarterly_retrain")
    summary = {"stage": "quarterly_retrain", "run_id": run_id, "model_type": config.MODEL_TYPE}
    try:
        clean_df = load_curated()
        if clean_df.empty:
            raise RuntimeError("No curated data. Run bootstrap_local.py or score_monthly.py first.")

        available, coverage = latest_ready_snapshot(clean_df)
        needed = config.N_TRAIN_SNAPSHOTS + config.N_VAL_SNAPSHOTS + config.N_TEST_SNAPSHOTS
        champion, champion_meta = load_champion(config.MODEL_TYPE)
        last_used = int(champion_meta.get("latest_snapshot_used", 0)) if champion_meta else 0
        summary.update({"available_labeled_snapshots": available, "last_snapshot_used": last_used})

        if available < needed:
            summary["retrain_skipped"] = f"Need {needed} fully labeled snapshots; only {available} available."
            log_run(summary)
            return summary
        if not force and champion is not None and available <= last_used:
            summary["retrain_skipped"] = f"No new labeled snapshot: available={available}, current champion already used={last_used}."
            log_run(summary)
            return summary

        bundle = build_rolling_train_val_test(
            clean_df,
            config.N_TRAIN_SNAPSHOTS,
            config.N_VAL_SNAPSHOTS,
            config.N_TEST_SNAPSHOTS,
            step_months=config.SNAPSHOT_STEP_MONTHS,
            data_through_date=coverage,
            end_snapshot=available,
        )
        summary["snapshot_split"] = {
            "train": bundle.train_snapshots,
            "val": bundle.val_snapshots,
            "test": bundle.test_snapshots,
        }

        if config.RETUNE_ON_RETRAIN:
            tuner = TUNERS[config.MODEL_TYPE]
            challenger, tune_result, _ = tuner(
                bundle.train.X,
                bundle.train.y,
                bundle.val.X,
                bundle.val.y,
                groups_train=bundle.train.customer_ids,
                snapshot_ids_train=bundle.train.snapshot_ids,
            )
            params = tune_result["params"]
            cv_auc = float(tune_result["cv_roc_auc"])
        else:
            challenger, params = build_production_candidate(config.MODEL_TYPE)
            cv_auc = temporal_cv_roc_auc(
                challenger, bundle.train.X, bundle.train.y, bundle.train.snapshot_ids
            )
            challenger.fit(bundle.train.X, bundle.train.y)

        threshold, _ = find_best_threshold(
            challenger, bundle.val.X, bundle.val.y, metric=config.THRESHOLD_METRIC
        )
        challenger_val = evaluate_fitted_model(
            challenger, bundle.val.X, bundle.val.y,
            label="Challenger validation", threshold=threshold,
        )
        challenger_val["cv_roc_auc"] = cv_auc

        if champion is None:
            promote, reason = True, "No existing champion; promoting first production model."
        else:
            champion_X_val = align_features(bundle.val.X, champion_meta["feature_columns"])
            champion_threshold = float(champion_meta["metrics"]["threshold"])
            champion_val = evaluate_fitted_model(
                champion, champion_X_val, bundle.val.y,
                label="Frozen champion on current validation", threshold=champion_threshold,
            )
            promote, reason = should_promote_challenger(
                challenger_val,
                champion_val,
                metric=config.PROMOTION_METRIC,
                max_metric_regression=config.PROMOTION_MAX_REGRESSION,
                max_recall_regression=config.PROMOTION_MAX_RECALL_REGRESSION,
                max_brier_regression=config.PROMOTION_MAX_BRIER_REGRESSION,
            )
            summary["champion_validation"] = champion_val

        summary["challenger_validation"] = challenger_val
        summary["promotion_decision"] = reason

        if promote:
            test_metrics = evaluate_fitted_model(
                challenger, bundle.test.X, bundle.test.y,
                label="Promoted challenger held-out test", threshold=threshold,
            )
            metrics_for_registry = dict(challenger_val)
            metrics_for_registry["threshold"] = threshold
            sample_n = min(config.DRIFT_BASELINE_SAMPLE_SIZE, len(bundle.train.X))
            sample = bundle.train.X.sample(sample_n, random_state=config.RANDOM_STATE)
            previous_version = champion_meta["version"] if champion_meta else None
            version = save_model(
                challenger,
                config.MODEL_TYPE,
                metrics=metrics_for_registry,
                params=params,
                training_snapshots=bundle.train_snapshots,
                validation_snapshots=bundle.val_snapshots,
                test_snapshots=bundle.test_snapshots,
                feature_columns=list(bundle.train.X.columns),
                latest_snapshot_used=bundle.latest_snapshot,
                training_feature_sample=sample,
                prediction_reference_probs=challenger.predict_proba(bundle.val.X)[:, 1],
                extra_metadata={
                    "test_metrics": test_metrics,
                    "data_through_date": str(coverage.date()) if coverage is not None else None,
                    "threshold_metric": config.THRESHOLD_METRIC,
                    "retuned": bool(config.RETUNE_ON_RETRAIN),
                    "promotion_reason": reason,
                    "replaced_champion_version": previous_version,
                },
            )
            set_current_version(config.MODEL_TYPE, version, preserve_previous=True)
            summary["previous_model_version"] = previous_version
            summary["active_model_version"] = version
            summary["test_metrics"] = test_metrics
        else:
            summary["active_model_version"] = champion_meta["version"]

        log_run(summary)
        return summary
    except Exception as exc:
        summary.update({"status": "FAILED", "error": str(exc)})
        log_run(summary)
        alert(str(exc), "error")
        raise


if __name__ == "__main__":
    main()
