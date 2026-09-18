"""Local immutable raw-file archive and ingestion idempotency manifest."""
from __future__ import annotations
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path | None = None) -> list[dict]:
    path = config.INGESTION_MANIFEST_PATH if path is None else Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_manifest(record: dict, path: Path | None = None) -> None:
    path = config.INGESTION_MANIFEST_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def find_successful_ingestion(checksum: str, snapshot_date: pd.Timestamp) -> dict | None:
    target = str(pd.Timestamp(snapshot_date).date())
    for row in reversed(read_manifest()):
        if row.get("status") == "SUCCESS" and row.get("sha256") == checksum and row.get("snapshot_date") == target:
            return row
    return None


def archive_raw_file(source: str | Path, snapshot_date: pd.Timestamp, checksum: str) -> Path:
    """Copy the exact delivered file into an immutable month folder."""
    source = Path(source).resolve()
    batch_end = pd.Timestamp(snapshot_date)
    batch_month = (batch_end - pd.Timedelta(days=1)).strftime("%Y-%m")
    target_dir = config.RAW_DATA_DIR / batch_month
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        if sha256_file(target) == checksum:
            return target
        target = target_dir / f"{source.stem}_{checksum[:8]}{source.suffix}"
    shutil.copy2(source, target)
    return target


def latest_data_through_date() -> pd.Timestamp | None:
    dates = [
        pd.Timestamp(r["snapshot_date"])
        for r in read_manifest()
        if r.get("status") == "SUCCESS" and r.get("snapshot_date")
    ]
    return max(dates) if dates else None
