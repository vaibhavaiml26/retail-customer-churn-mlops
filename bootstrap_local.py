"""One-time local bootstrap: create curated history and train the first champion."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import config
from data_loader import clean_transactions, load_transactions
from data_validation import validate_schema
from pipeline_common import update_curated_store
from retrain_quarterly import main as retrain


def main(files: list[str]):
    raw_bootstrap = config.RAW_DATA_DIR / "bootstrap"
    raw_bootstrap.mkdir(parents=True, exist_ok=True)
    for file in files:
        src = Path(file).resolve()
        dst = raw_bootstrap / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    combined = load_transactions(files)
    structural = validate_schema(combined)
    if not structural.passed:
        raise ValueError(structural.report())
    clean = clean_transactions(combined)
    # Bootstrap replaces an empty local curated store. Refuse to silently mix
    # a second historical bootstrap with already-operational production data.
    if config.CURATED_DATA_PATH.exists():
        raise FileExistsError(
            f"{config.CURATED_DATA_PATH} already exists. Delete/reset the local environment intentionally before bootstrapping again."
        )
    update_curated_store(clean)
    return retrain(force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=config.DATA_FILES)
    args = parser.parse_args()
    main(args.files)
