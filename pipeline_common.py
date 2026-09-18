"""Shared local-production I/O, validation, logging and feature-schema utilities."""
from __future__ import annotations
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import config
from data_loader import clean_transactions
from data_validation import compute_monthly_baseline, detect_volume_anomalies, validate_schema
from ingestion_manifest import append_manifest, archive_raw_file, find_successful_ingestion, sha256_file

RUN_LOG_PATH = config.MONITORING_DIR / "run_log.jsonl"
MONITORING_HISTORY_PATH = config.MONITORING_DIR / "monitoring_history.csv"
MODEL_PERFORMANCE_HISTORY_PATH = config.MONITORING_DIR / "model_performance_history.csv"
CALIBRATION_HISTORY_PATH = config.MONITORING_DIR / "calibration_history.csv"
SHADOW_COMPARISON_HISTORY_PATH = config.MONITORING_DIR / "shadow_model_comparison_history.csv"


def new_run_id(stage: str) -> str:
    return f"{stage}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def alert(message: str, severity: str = "warning"):
    print(f"[ALERT:{severity.upper()}] {message}", file=sys.stderr)


def log_run(entry: dict):
    config.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(entry)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


def append_history_row(path: Path, row: dict):
    """Append a small local history row while allowing the schema to evolve.

    Monitoring files gain columns over time (for example calibration and
    shadow-model fields). Rewriting the small local CSV atomically is safer
    than appending a row with a different column count under an old header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        columns = list(existing.columns) + [c for c in incoming.columns if c not in existing.columns]
        existing = existing.reindex(columns=columns)
        incoming = incoming.reindex(columns=columns)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)


def load_curated() -> pd.DataFrame:
    if not config.CURATED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.CURATED_DATA_PATH)


def update_curated_store(new_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Append only cleaned data; exact reruns remain idempotent via deduplication."""
    config.CURATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    historical = load_curated()
    combined = pd.concat([historical, new_clean_df], ignore_index=True) if not historical.empty else new_clean_df.copy()
    combined = combined.drop_duplicates().sort_values("InvoiceDate").reset_index(drop=True)
    tmp = config.CURATED_DATA_PATH.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    tmp.replace(config.CURATED_DATA_PATH)
    return combined


def _validate_batch_dates(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> None:
    """A monthly file must belong to the month immediately before snapshot_date."""
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    batch_start = snapshot_date - pd.DateOffset(months=1)
    bad = df[(df["InvoiceDate"] < batch_start) | (df["InvoiceDate"] >= snapshot_date)]
    if not bad.empty:
        raise ValueError(
            f"Monthly batch contains {len(bad)} rows outside [{batch_start.date()}, {snapshot_date.date()})."
        )


def ingest_validate_curate(new_file: str, snapshot_date: pd.Timestamp) -> tuple[pd.DataFrame, dict, bool]:
    """Archive, validate, clean, and append one complete monthly batch.

    Returns ``(full_curated_history, ingestion_record, was_new_ingestion)``.
    Re-running the exact same file/snapshot is a safe no-op at ingestion level.
    """
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    checksum = sha256_file(new_file)
    existing = find_successful_ingestion(checksum, snapshot_date)
    if existing:
        curated = load_curated()
        if curated.empty:
            raise RuntimeError("Manifest says batch was ingested but curated store is missing.")
        return curated, existing, False

    archived = archive_raw_file(new_file, snapshot_date, checksum)
    record = {
        "source_file": str(Path(new_file).resolve()),
        "archived_file": str(archived),
        "sha256": checksum,
        "snapshot_date": str(snapshot_date.date()),
    }
    try:
        raw = pd.read_csv(archived)
        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], format=config.DATE_FORMAT)
        structural = validate_schema(raw)
        if not structural.passed:
            raise ValueError(structural.report())
        _validate_batch_dates(raw, snapshot_date)

        clean = clean_transactions(raw)
        historical = load_curated()
        if not historical.empty:
            baseline = compute_monthly_baseline(historical)
            volume = detect_volume_anomalies(clean, baseline)
            if not volume.passed:
                raise ValueError(volume.report())
            if volume.warnings:
                alert(volume.report(), "warning")

        combined = update_curated_store(clean)
        record.update({"status": "SUCCESS", "raw_rows": len(raw), "clean_rows": len(clean)})
        append_manifest(record)
        return combined, record, True
    except Exception as exc:
        record.update({"status": "FAILED", "error": str(exc)})
        append_manifest(record)
        raise


def align_features(X: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in expected_columns if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_columns]
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. missing={missing}, extra={extra}")
    return X.loc[:, expected_columns]
