"""Resolve delayed labels, calibration, and official-vs-shadow production metrics."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import config
from drift_monitoring import evaluate_delayed_predictions
from pipeline_common import (
    CALIBRATION_HISTORY_PATH,
    MODEL_PERFORMANCE_HISTORY_PATH,
    SHADOW_COMPARISON_HISTORY_PATH,
    append_history_row,
)


def build_actual_outcomes(
    clean_transactions_df: pd.DataFrame,
    customer_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Label one scored cohort using future *sales* only."""
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    outcome_end = snapshot_date + pd.DateOffset(months=config.OUTCOME_PERIOD_MONTHS)
    future_sales = clean_transactions_df[
        (clean_transactions_df["InvoiceDate"] >= snapshot_date)
        & (clean_transactions_df["InvoiceDate"] < outcome_end)
        & (clean_transactions_df["Quantity"] > 0)
    ]
    active = set(future_sales["CustomerID"].dropna().unique())
    out = pd.DataFrame({"CustomerID": customer_ids.drop_duplicates().values})
    out["snapshot_date"] = snapshot_date
    out["outcome_start"] = snapshot_date
    out["outcome_end"] = outcome_end
    out["inactive_90d"] = (~out["CustomerID"].isin(active)).astype(int)
    out["resolved_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _load_labels() -> pd.DataFrame:
    if not config.LABELS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(config.LABELS_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _save_labels(labels: pd.DataFrame):
    config.LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.drop_duplicates(["CustomerID", "snapshot_date"], keep="last")
    labels.to_parquet(config.LABELS_PATH, index=False)


def _evaluated_prediction_sets() -> set[tuple[str, str, str]]:
    path = MODEL_PERFORMANCE_HISTORY_PATH
    if not path.exists():
        return set()
    hist = pd.read_csv(path)
    if "snapshot_date" not in hist or "model_version" not in hist:
        return set()
    if "prediction_role" not in hist:
        hist["prediction_role"] = "official"
    return set(zip(
        hist["snapshot_date"].astype(str),
        hist["model_version"].astype(str),
        hist["prediction_role"].fillna("official").astype(str),
    ))


def _prediction_files() -> list[tuple[str, object]]:
    files = [("official", p) for p in sorted(config.PREDICTIONS_DIR.glob("snapshot_*.parquet"))]
    if config.SHADOW_PREDICTIONS_DIR.exists():
        files.extend(("shadow", p) for p in sorted(config.SHADOW_PREDICTIONS_DIR.glob("snapshot_*.parquet")))
    return files


def _append_calibration_rows(snapshot_key, outcome_end, model_version, prediction_role, calibration_bins):
    for row in calibration_bins:
        append_history_row(CALIBRATION_HISTORY_PATH, {
            "snapshot_date": snapshot_key,
            "outcome_end": str(outcome_end.date()),
            "model_version": model_version,
            "prediction_role": prediction_role,
            **row,
        })


def _existing_shadow_comparisons() -> set[str]:
    if not SHADOW_COMPARISON_HISTORY_PATH.exists():
        return set()
    hist = pd.read_csv(SHADOW_COMPARISON_HISTORY_PATH)
    return set(hist["snapshot_date"].astype(str)) if "snapshot_date" in hist else set()


def _write_new_shadow_comparisons():
    """Compare official and shadow models only when both are resolved on the same cohort."""
    if not MODEL_PERFORMANCE_HISTORY_PATH.exists():
        return
    hist = pd.read_csv(MODEL_PERFORMANCE_HISTORY_PATH)
    if "prediction_role" not in hist:
        hist["prediction_role"] = "official"
    done = _existing_shadow_comparisons()

    for snapshot_key, cohort in hist.groupby("snapshot_date"):
        key = str(snapshot_key)
        if key in done:
            continue
        official = cohort[cohort["prediction_role"] == "official"]
        shadow = cohort[cohort["prediction_role"] == "shadow"]
        if official.empty or shadow.empty:
            continue
        off = official.iloc[-1]
        sh = shadow.iloc[-1]
        row = {
            "snapshot_date": key,
            "official_model_version": off["model_version"],
            "shadow_model_version": sh["model_version"],
            "official_roc_auc": off.get("roc_auc"),
            "shadow_roc_auc": sh.get("roc_auc"),
            "roc_auc_delta_official_minus_shadow": off.get("roc_auc") - sh.get("roc_auc"),
            "official_pr_auc": off.get("pr_auc"),
            "shadow_pr_auc": sh.get("pr_auc"),
            "pr_auc_delta_official_minus_shadow": off.get("pr_auc") - sh.get("pr_auc"),
            "official_recall": off.get("recall"),
            "shadow_recall": sh.get("recall"),
            "recall_delta_official_minus_shadow": off.get("recall") - sh.get("recall"),
            "official_brier_score": off.get("brier_score"),
            "shadow_brier_score": sh.get("brier_score"),
            "brier_delta_official_minus_shadow": off.get("brier_score") - sh.get("brier_score"),
        }
        append_history_row(SHADOW_COMPARISON_HISTORY_PATH, row)
        done.add(key)


def resolve_matured_predictions(clean_transactions_df: pd.DataFrame, current_snapshot_date: pd.Timestamp) -> list[dict]:
    """Resolve every official and shadow prediction set whose outcome window has elapsed."""
    current_snapshot_date = pd.Timestamp(current_snapshot_date).normalize()
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    labels = _load_labels()
    evaluated = _evaluated_prediction_sets()
    new_metric_rows = []

    for inferred_role, path in _prediction_files():
        preds = pd.read_parquet(path)
        if preds.empty or "snapshot_date" not in preds:
            continue
        preds["snapshot_date"] = pd.to_datetime(preds["snapshot_date"])
        snapshot_date = pd.Timestamp(preds["snapshot_date"].iloc[0]).normalize()
        outcome_end = snapshot_date + pd.DateOffset(months=config.OUTCOME_PERIOD_MONTHS)
        if outcome_end > current_snapshot_date:
            continue

        snapshot_key = str(snapshot_date.date())
        model_version = str(preds["model_version"].iloc[0]) if "model_version" in preds else "unknown"
        prediction_role = (
            str(preds["prediction_role"].iloc[0])
            if "prediction_role" in preds
            else inferred_role
        )

        existing = labels[labels["snapshot_date"] == snapshot_date] if not labels.empty else pd.DataFrame()
        if existing.empty:
            cohort_labels = build_actual_outcomes(clean_transactions_df, preds["CustomerID"], snapshot_date)
            labels = pd.concat([labels, cohort_labels], ignore_index=True) if not labels.empty else cohort_labels
            _save_labels(labels)
            existing = cohort_labels

        eval_key = (snapshot_key, model_version, prediction_role)
        if eval_key in evaluated:
            continue

        metrics = evaluate_delayed_predictions(
            preds,
            existing,
            calibration_bins=config.CALIBRATION_BINS,
        )
        calibration_bins = metrics.pop("calibration_bins", [])
        row = {
            "snapshot_date": snapshot_key,
            "outcome_end": str(outcome_end.date()),
            "model_version": model_version,
            "prediction_role": prediction_role,
            **metrics,
        }
        append_history_row(MODEL_PERFORMANCE_HISTORY_PATH, row)
        _append_calibration_rows(
            snapshot_key, outcome_end, model_version, prediction_role, calibration_bins
        )
        new_metric_rows.append(row)
        evaluated.add(eval_key)

    _write_new_shadow_comparisons()
    return new_metric_rows
