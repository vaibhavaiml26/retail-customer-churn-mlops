"""Append-only local model registry with CURRENT/PREVIOUS model pointers."""
from __future__ import annotations
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import sklearn
import xgboost
import config


def _registry_dir(model_type: str) -> Path:
    return config.REGISTRY_ROOT / model_type


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_%f")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def runtime_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "git_commit": _git_commit(),
    }


def save_model(
    model,
    model_type: str,
    metrics: dict,
    params: dict,
    training_snapshots: list[int],
    feature_columns: list[str],
    latest_snapshot_used: int,
    validation_snapshots: list[int] | None = None,
    test_snapshots: list[int] | None = None,
    extra_metadata: dict | None = None,
    training_feature_sample: pd.DataFrame | None = None,
    prediction_reference_probs=None,
) -> str:
    model_dir = _registry_dir(model_type)
    version = new_version_id()
    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, version_dir / "model.joblib")

    metadata = {
        "version": version,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "params": params,
        "feature_columns": list(feature_columns),
        "training_snapshots": list(training_snapshots),
        "validation_snapshots": list(validation_snapshots or []),
        "test_snapshots": list(test_snapshots or []),
        "latest_snapshot_used": int(latest_snapshot_used),
        **runtime_metadata(),
        **(extra_metadata or {}),
    }

    if training_feature_sample is not None and not training_feature_sample.empty:
        sample_path = version_dir / "training_feature_sample.parquet"
        training_feature_sample.to_parquet(sample_path, index=False)
        metadata["training_feature_sample_path"] = str(sample_path)
        probs = model.predict_proba(training_feature_sample)[:, 1]
        pred_path = version_dir / "training_prediction_sample.parquet"
        pd.DataFrame({"churn_probability": probs}).to_parquet(pred_path, index=False)
        metadata["training_prediction_sample_path"] = str(pred_path)

    if prediction_reference_probs is not None:
        pred_ref_path = version_dir / "prediction_reference_sample.parquet"
        pd.DataFrame({"churn_probability": prediction_reference_probs}).to_parquet(pred_ref_path, index=False)
        metadata["prediction_reference_sample_path"] = str(pred_ref_path)

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return version


def load_model(model_type: str, version: str):
    version_dir = _registry_dir(model_type) / version
    model = joblib.load(version_dir / "model.joblib")
    with open(version_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def _read_pointer(model_type: str, pointer_name: str) -> str | None:
    pointer = _registry_dir(model_type) / pointer_name
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def _write_pointer(model_type: str, pointer_name: str, version: str):
    model_dir = _registry_dir(model_type)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / version / "model.joblib").exists():
        raise FileNotFoundError(f"Model version {version} does not exist.")
    tmp = model_dir / f"{pointer_name}.tmp"
    tmp.write_text(version, encoding="utf-8")
    tmp.replace(model_dir / pointer_name)


def get_current_version(model_type: str) -> str | None:
    return _read_pointer(model_type, "CURRENT")


def get_previous_version(model_type: str) -> str | None:
    """Return the immediately previous champion.

    New registries maintain an explicit PREVIOUS pointer. For an existing
    registry created by the earlier project version, fall back to the newest
    registered version other than CURRENT so shadow scoring works immediately.
    """
    current = get_current_version(model_type)
    explicit = _read_pointer(model_type, "PREVIOUS")
    if explicit and explicit != current:
        return explicit

    if current is None:
        return None
    candidates = [m["version"] for m in list_versions(model_type) if m.get("version") != current]
    return candidates[0] if candidates else None


def set_current_version(model_type: str, version: str, preserve_previous: bool = True):
    """Atomically update CURRENT and preserve the replaced champion as PREVIOUS."""
    old_current = get_current_version(model_type)
    if old_current == version:
        return
    if preserve_previous and old_current is not None:
        _write_pointer(model_type, "PREVIOUS", old_current)
    _write_pointer(model_type, "CURRENT", version)


def load_champion(model_type: str):
    version = get_current_version(model_type)
    return (None, None) if version is None else load_model(model_type, version)


def load_previous_champion(model_type: str):
    version = get_previous_version(model_type)
    return (None, None) if version is None else load_model(model_type, version)


def list_versions(model_type: str) -> list[dict]:
    model_dir = _registry_dir(model_type)
    if not model_dir.exists():
        return []
    rows = []
    for version_dir in sorted(model_dir.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return rows
