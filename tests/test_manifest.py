import pandas as pd
import config
from ingestion_manifest import append_manifest, find_successful_ingestion, read_manifest


def test_ingestion_manifest_detects_successful_rerun(tmp_path, monkeypatch):
    path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(config, "INGESTION_MANIFEST_PATH", path)
    record = {
        "status": "SUCCESS",
        "sha256": "abc",
        "snapshot_date": "2027-10-01",
    }
    append_manifest(record)
    assert len(read_manifest()) == 1
    found = find_successful_ingestion("abc", pd.Timestamp("2027-10-01"))
    assert found is not None
