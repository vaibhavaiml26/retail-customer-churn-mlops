"""
Azure ML quarterly retraining job for the retail churn project.

This job reconstructs all transaction history available before an explicit
as-of date, builds the newest fully-resolved rolling train/validation/test
snapshots, trains a frozen-parameter XGBoost challenger, compares it with the
registered champion on the SAME validation snapshot, applies promotion
safety guardrails, and writes the challenger + decision as Azure ML outputs.

Registration is deliberately NOT performed inside this compute job.  The
local submitter performs registration only after reading promotion_decision.json.
That keeps Azure credentials out of the training container.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project modules live one directory above azure/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
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
from xgboost import XGBClassifier

import config
from data_loader import clean_transactions, load_transactions
from data_validation import validate_schema
from feature_engineering import generate_snapshot_dataset, max_available_snapshots
from models import find_best_threshold


def _resolve_csv_files(input_path: str, label: str) -> list[str]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{label} input is not a CSV: {path}")
        files = [str(path)]
    else:
        files = sorted(
            str(p)
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() == ".csv"
        )

    if not files:
        raise ValueError(f"No CSV files found under {label} input: {path}")

    print(f"{label}: found {len(files)} CSV file(s)")
    for f in files:
        print(f"  {f}")
    return files


def _resolve_model_files(model_path: str) -> tuple[Path, Path]:
    root = Path(model_path)
    if not root.exists():
        raise FileNotFoundError(f"Champion model input does not exist: {root}")

    model_files = list(root.rglob("model.joblib")) if root.is_dir() else []
    metadata_files = list(root.rglob("metadata.json")) if root.is_dir() else []

    if len(model_files) != 1:
        raise ValueError(
            f"Expected exactly one model.joblib under champion model input; found {len(model_files)}"
        )
    if len(metadata_files) != 1:
        raise ValueError(
            f"Expected exactly one metadata.json under champion model input; found {len(metadata_files)}"
        )
    return model_files[0], metadata_files[0]


def _build_split(
    clean_df: pd.DataFrame, snapshot_numbers: list[int], step_months: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    group_parts: list[pd.Series] = []

    for snapshot_no in snapshot_numbers:
        X, y, groups = generate_snapshot_dataset(clean_df, snapshot_no, step_months)
        X_parts.append(X)
        y_parts.append(y)
        group_parts.append(groups)

    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
        pd.concat(group_parts, ignore_index=True),
    )


def _build_xgb(params: dict) -> XGBClassifier:
    final_params = dict(params)
    final_params.setdefault("random_state", getattr(config, "RANDOM_STATE", 42))
    final_params.setdefault("eval_metric", "logloss")
    return XGBClassifier(**final_params)


def _evaluate_fitted(model, X, y, threshold: float, label: str) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    metrics = {
        "label": label,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "mean_probability": float(np.mean(probs)),
        "actual_churn_rate": float(np.mean(y)),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n--- {label} ---")
    for key in [
        "threshold", "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "brier_score", "mean_probability",
        "actual_churn_rate",
    ]:
        print(f"{key}: {metrics[key]:.6f}")
    print("confusion_matrix:")
    print(cm)
    return metrics


def _forward_cv_roc_auc(
    clean_df: pd.DataFrame,
    train_snapshots: list[int],
    step_months: int,
    params: dict,
) -> tuple[float | None, list[dict]]:
    """Snapshot-aware forward CV: earlier snapshot(s) -> next snapshot."""
    rows: list[dict] = []
    if len(train_snapshots) < 2:
        return None, rows

    for i in range(1, len(train_snapshots)):
        fold_train = train_snapshots[:i]
        fold_val = [train_snapshots[i]]

        X_tr, y_tr, _ = _build_split(clean_df, fold_train, step_months)
        X_va, y_va, _ = _build_split(clean_df, fold_val, step_months)

        model = _build_xgb(params)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y_va, probs))
        row = {
            "train_snapshots": fold_train,
            "validation_snapshot": fold_val[0],
            "roc_auc": auc,
        }
        rows.append(row)
        print(
            f"Forward CV: train={fold_train} -> val={fold_val[0]} "
            f"ROC-AUC={auc:.6f}"
        )

    return float(np.mean([r["roc_auc"] for r in rows])), rows


def _promotion_guardrails(challenger: dict, champion: dict) -> tuple[bool, str, dict]:
    max_roc_reg = float(
        getattr(config, "PROMOTION_MAX_ROC_AUC_REGRESSION", 0.005)
    )
    max_recall_reg = float(
        getattr(config, "PROMOTION_MAX_RECALL_REGRESSION", 0.05)
    )
    max_brier_worsening = float(
        getattr(
            config,
            "PROMOTION_MAX_BRIER_WORSENING",
            getattr(config, "PROMOTION_MAX_BRIER_DEGRADATION", 0.02),
        )
    )

    roc_delta = challenger["roc_auc"] - champion["roc_auc"]
    recall_delta = challenger["recall"] - champion["recall"]
    brier_delta = challenger["brier_score"] - champion["brier_score"]

    checks = {
        "max_roc_auc_regression": max_roc_reg,
        "max_recall_regression": max_recall_reg,
        "max_brier_worsening": max_brier_worsening,
        "roc_auc_delta": float(roc_delta),
        "recall_delta": float(recall_delta),
        "brier_delta": float(brier_delta),
        "roc_auc_pass": bool(roc_delta >= -max_roc_reg),
        "recall_pass": bool(recall_delta >= -max_recall_reg),
        "brier_pass": bool(brier_delta <= max_brier_worsening),
    }
    promote = checks["roc_auc_pass"] and checks["recall_pass"] and checks["brier_pass"]

    if promote:
        reason = (
            "PROMOTE: challenger passed all guardrails. "
            f"ROC-AUC delta={roc_delta:+.4f} (limit -{max_roc_reg:.4f}), "
            f"recall delta={recall_delta:+.4f} (limit -{max_recall_reg:.4f}), "
            f"Brier delta={brier_delta:+.4f} (limit +{max_brier_worsening:.4f})."
        )
    else:
        failed = [
            name
            for name, passed in [
                ("ROC-AUC", checks["roc_auc_pass"]),
                ("recall", checks["recall_pass"]),
                ("Brier", checks["brier_pass"]),
            ]
            if not passed
        ]
        reason = (
            f"KEEP CHAMPION: challenger failed guardrail(s): {', '.join(failed)}. "
            f"ROC-AUC delta={roc_delta:+.4f}, recall delta={recall_delta:+.4f}, "
            f"Brier delta={brier_delta:+.4f}."
        )

    return promote, reason, checks


def _get_threshold(metadata: dict) -> float:
    candidates = [
        metadata.get("threshold"),
        metadata.get("metrics", {}).get("threshold") if isinstance(metadata.get("metrics"), dict) else None,
        metadata.get("validation_metrics", {}).get("threshold")
        if isinstance(metadata.get("validation_metrics"), dict)
        else None,
    ]
    for value in candidates:
        if value is not None:
            return float(value)
    raise ValueError("Champion metadata does not contain a threshold.")



def _write_no_challenger_marker(challenger_model_output: str | Path, decision: dict) -> None:
    """Ensure the optional challenger output is non-empty when retraining is skipped.

    Azure ML pipeline outputs are storage-backed.  A small marker keeps the
    URI-folder output materialized without pretending that a trained model
    exists.  The submitter still registers a model only when promote=true.
    """
    model_dir = Path(challenger_model_output)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "NO_CHALLENGER.json").write_text(
        json.dumps(
            {
                "challenger_created": False,
                "retrain_skipped": bool(decision.get("retrain_skipped")),
                "promote": bool(decision.get("promote")),
                "reason": decision.get("reason"),
                "as_of_date": decision.get("as_of_date"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

from datetime import datetime, timezone
import pandas as pd


'''def resolve_as_of_date(value: str) -> pd.Timestamp:
    if value.strip().upper() == "AUTO":
        today = datetime.now(timezone.utc).date()
        first_day = today.replace(day=1)
        return pd.Timestamp(first_day)

    return pd.Timestamp(value)'''


def main(
    bootstrap_data: str,
    replay_data: str,
    champion_model_path: str,
    champion_model_ref: str,
    challenger_model_output: str,
    decision_output: str,
    as_of_date: str,
    n_train: int,
    n_val: int,
    n_test: int,
    step_months: int,
) -> dict:
    cutoff = pd.Timestamp(as_of_date)

    print("Azure quarterly churn retraining")
    print("=" * 70)
    print("As-of date:", cutoff.date())
    print("Champion model:", champion_model_ref)
    print(f"Split: train={n_train}, val={n_val}, test={n_test}, step={step_months} months")

    bootstrap_files = _resolve_csv_files(bootstrap_data, "Bootstrap")
    replay_files = _resolve_csv_files(replay_data, "Replay")

    bootstrap_df = load_transactions(bootstrap_files)
    replay_all = load_transactions(replay_files)
    replay_used = replay_all[replay_all["InvoiceDate"] < cutoff].copy()

    if replay_used.empty:
        raise ValueError(f"No replay transactions exist before as-of date {cutoff.date()}.")

    if replay_used["InvoiceDate"].min() <= bootstrap_df["InvoiceDate"].max():
        raise ValueError(
            "Bootstrap and replay data overlap in time. Refusing to double-count history. "
            f"bootstrap_max={bootstrap_df['InvoiceDate'].max()}, "
            f"replay_min_used={replay_used['InvoiceDate'].min()}"
        )

    history = pd.concat([bootstrap_df, replay_used], ignore_index=True)
    history = history[history["InvoiceDate"] < cutoff].copy()
    history = history.sort_values("InvoiceDate").reset_index(drop=True)

    print("\nRetraining-history summary")
    print("-" * 70)
    print(f"Bootstrap rows available: {len(bootstrap_df):,}")
    print(f"Replay rows used:         {len(replay_used):,}")
    print(f"Combined raw rows:        {len(history):,}")
    print(
        f"Combined date range:      {history['InvoiceDate'].min()} -> "
        f"{history['InvoiceDate'].max()}"
    )

    validation = validate_schema(history)
    print(validation.report())
    if not validation.passed:
        raise ValueError("Combined retraining history failed structural validation.")

    clean_df = clean_transactions(history)
    print(f"Clean rows:               {len(clean_df):,}")

    total_needed = n_train + n_val + n_test
    available = max_available_snapshots(clean_df, step_months)
    print(f"Fully resolved snapshots available: {available}")
    print(f"Snapshots needed for split:         {total_needed}")

    decision_dir = Path(decision_output)
    decision_dir.mkdir(parents=True, exist_ok=True)

    # Load the current champion before deciding whether a retrain is due.
    # Besides giving us its threshold/model for the eventual comparison, its
    # metadata tells us the newest fully-resolved snapshot already consumed
    # by the champion. This makes the job safe to invoke every month: when no
    # newer labeled snapshot has matured, it exits cheaply instead of
    # retraining the exact same challenger again.
    champion_model_file, champion_metadata_file = _resolve_model_files(champion_model_path)
    champion = joblib.load(champion_model_file)
    champion_metadata = json.loads(champion_metadata_file.read_text(encoding="utf-8"))
    champion_threshold = _get_threshold(champion_metadata)

    champion_latest_snapshot = champion_metadata.get("latest_snapshot_available")
    if champion_latest_snapshot is None:
        champion_latest_snapshot = champion_metadata.get("latest_snapshot_used")
    if champion_latest_snapshot is not None:
        champion_latest_snapshot = int(champion_latest_snapshot)

    if available < total_needed:
        decision = {
            "stage": "azure_quarterly_retrain",
            "retrain_skipped": True,
            "promote": False,
            "reason": (
                f"Not enough fully resolved snapshots: available={available}, "
                f"need={total_needed}, step_months={step_months}."
            ),
            "as_of_date": str(cutoff.date()),
            "champion_model_ref": champion_model_ref,
        }
        (decision_dir / "promotion_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        _write_no_challenger_marker(challenger_model_output, decision)
        print(json.dumps(decision, indent=2))
        return decision

    if champion_latest_snapshot is not None and available <= champion_latest_snapshot:
        decision = {
            "stage": "azure_quarterly_retrain",
            "retrain_skipped": True,
            "promote": False,
            "reason": (
                "No new fully resolved snapshot beyond the current champion: "
                f"available={available}, champion_latest_snapshot={champion_latest_snapshot}."
            ),
            "as_of_date": str(cutoff.date()),
            "champion_model_ref": champion_model_ref,
            "available_snapshots": int(available),
            "champion_latest_snapshot": int(champion_latest_snapshot),
        }
        (decision_dir / "promotion_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        _write_no_challenger_marker(challenger_model_output, decision)
        print("\nRetraining readiness check")
        print("-" * 70)
        print(decision["reason"])
        print(json.dumps(decision, indent=2))
        return decision

    window_start = available - total_needed + 1
    train_snapshots = list(range(window_start, window_start + n_train))
    val_snapshots = list(range(train_snapshots[-1] + 1, train_snapshots[-1] + 1 + n_val))
    test_snapshots = list(range(val_snapshots[-1] + 1, val_snapshots[-1] + 1 + n_test))

    print("\nRolling snapshot split")
    print("-" * 70)
    print("Train snapshots:", train_snapshots)
    print("Validation snapshots:", val_snapshots)
    print("Test snapshots:", test_snapshots)

    X_train, y_train, groups_train = _build_split(clean_df, train_snapshots, step_months)
    X_val, y_val, groups_val = _build_split(clean_df, val_snapshots, step_months)
    X_test, y_test, groups_test = _build_split(clean_df, test_snapshots, step_months)

    print(f"Train: {X_train.shape}, churn={y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, churn={y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, churn={y_test.mean():.4f}")

    expected_features = champion_metadata.get("feature_columns")
    if expected_features:
        actual_features = list(X_train.columns)
        if actual_features != expected_features:
            raise ValueError(
                "Feature schema mismatch against champion metadata. "
                f"expected={expected_features}, actual={actual_features}"
            )

    params = dict(
        getattr(
            config,
            "BEST_XGB_PARAMS",
            {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.6,
                "colsample_bytree": 0.8,
            },
        )
    )

    print("\nChallenger frozen parameters")
    print("-" * 70)
    print(json.dumps(params, indent=2))

    cv_roc_auc, cv_folds = _forward_cv_roc_auc(
        clean_df, train_snapshots, step_months, params
    )
    if cv_roc_auc is not None:
        print(f"Mean forward CV ROC-AUC: {cv_roc_auc:.6f}")

    challenger = _build_xgb(params)
    challenger.fit(X_train, y_train)

    threshold_metric = getattr(config, "THRESHOLD_METRIC", "youden_j")
    challenger_threshold, threshold_sweep = find_best_threshold(
        challenger, X_val, y_val, metric=threshold_metric
    )

    challenger_val = _evaluate_fitted(
        challenger, X_val, y_val, challenger_threshold, "Challenger on validation"
    )
    champion_val = _evaluate_fitted(
        champion, X_val, y_val, champion_threshold, "Champion on same validation"
    )

    promote, reason, guardrails = _promotion_guardrails(challenger_val, champion_val)
    print("\nPromotion decision")
    print("-" * 70)
    print(reason)

    challenger_test = None
    if promote:
        # Test is inspected only after the promotion decision has already been made.
        challenger_test = _evaluate_fitted(
            challenger, X_test, y_test, challenger_threshold, "Promoted challenger on TEST"
        )

    model_dir = Path(challenger_model_output)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger, model_dir / "model.joblib")

    training_sample = X_train.sample(min(500, len(X_train)), random_state=42)
    training_sample.to_parquet(model_dir / "training_feature_sample.parquet", index=False)

    val_probs = challenger.predict_proba(X_val)[:, 1]
    pd.DataFrame({"churn_probability": val_probs}).to_parquet(
        model_dir / "prediction_reference_sample.parquet", index=False
    )
    threshold_sweep.to_csv(model_dir / "threshold_sweep.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": challenger.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    run_id = os.getenv("AZUREML_RUN_ID") or os.getenv("AZUREML_JOB_NAME")
    metadata = {
        "model_type": "xgboost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_run_id": run_id,
        "as_of_date": str(cutoff.date()),
        "champion_model_ref": champion_model_ref,
        "params": params,
        "threshold": float(challenger_threshold),
        "metrics": {**challenger_val, "threshold": float(challenger_threshold)},
        "validation_metrics": challenger_val,
        "test_metrics": challenger_test,
        "forward_cv_roc_auc": cv_roc_auc,
        "forward_cv_folds": cv_folds,
        "training_snapshots": train_snapshots,
        "validation_snapshots": val_snapshots,
        "test_snapshots": test_snapshots,
        "latest_snapshot_available": available,
        "snapshot_step_months": step_months,
        "feature_columns": list(X_train.columns),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "training_churn_rate": float(y_train.mean()),
        "validation_churn_rate": float(y_val.mean()),
        "test_churn_rate": float(y_test.mean()),
        "promotion": {
            "promote": bool(promote),
            "reason": reason,
            "guardrails": guardrails,
            "champion_validation_metrics": champion_val,
            "challenger_validation_metrics": challenger_val,
        },
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "history_date_min": str(history["InvoiceDate"].min()),
        "history_date_max": str(history["InvoiceDate"].max()),
        "raw_rows": int(len(history)),
        "clean_rows": int(len(clean_df)),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    decision = {
        "stage": "azure_quarterly_retrain",
        "retrain_skipped": False,
        "promote": bool(promote),
        "reason": reason,
        "as_of_date": str(cutoff.date()),
        "champion_model_ref": champion_model_ref,
        "challenger_threshold": float(challenger_threshold),
        "champion_threshold": float(champion_threshold),
        "train_snapshots": train_snapshots,
        "validation_snapshots": val_snapshots,
        "test_snapshots": test_snapshots,
        "forward_cv_roc_auc": cv_roc_auc,
        "challenger_validation_metrics": challenger_val,
        "champion_validation_metrics": champion_val,
        "challenger_test_metrics": challenger_test,
        "guardrails": guardrails,
        "azure_run_id": run_id,
    }
    (decision_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    print("\nAzure quarterly retraining completed successfully.")
    print(json.dumps(decision, indent=2, default=str))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-data", required=True)
    parser.add_argument("--replay-data", required=True)
    parser.add_argument("--champion-model", required=True)
    parser.add_argument("--champion-model-ref", required=True)
    parser.add_argument("--challenger-model-output", required=True)
    parser.add_argument("--decision-output", required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--n-train", type=int, default=2)
    parser.add_argument("--n-val", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=3)
    args = parser.parse_args()

    main(
        bootstrap_data=args.bootstrap_data,
        replay_data=args.replay_data,
        champion_model_path=args.champion_model,
        champion_model_ref=args.champion_model_ref,
        challenger_model_output=args.challenger_model_output,
        decision_output=args.decision_output,
        as_of_date=args.as_of_date,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        step_months=args.step_months,
    )
