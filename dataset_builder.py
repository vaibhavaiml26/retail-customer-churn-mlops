"""Build fixed or rolling temporal datasets from labeled customer snapshots."""
from dataclasses import dataclass
import pandas as pd

from config import FEATURE_PERIOD_MONTHS
from feature_engineering import generate_snapshot_dataset, max_available_snapshots, recommend_step_months


@dataclass
class SplitData:
    X: pd.DataFrame
    y: pd.Series
    customer_ids: pd.Series
    snapshot_ids: pd.Series


@dataclass
class DatasetBundle:
    train: SplitData
    val: SplitData
    test: SplitData
    train_snapshots: list[int]
    val_snapshots: list[int]
    test_snapshots: list[int]

    @property
    def latest_snapshot(self) -> int:
        return self.test_snapshots[-1]


def build_split(clean_transactions_df: pd.DataFrame, snapshot_numbers: list[int], step_months: int) -> SplitData:
    X_parts, y_parts, customer_parts, snapshot_parts = [], [], [], []
    for snapshot_no in snapshot_numbers:
        X, y, customer_ids = generate_snapshot_dataset(clean_transactions_df, snapshot_no, step_months)
        X_parts.append(X)
        y_parts.append(y)
        customer_parts.append(customer_ids.reset_index(drop=True))
        snapshot_parts.append(pd.Series([snapshot_no] * len(X), name="snapshot_id"))

    if not X_parts:
        raise ValueError("snapshot_numbers cannot be empty")

    return SplitData(
        X=pd.concat(X_parts, ignore_index=True),
        y=pd.concat(y_parts, ignore_index=True),
        customer_ids=pd.concat(customer_parts, ignore_index=True),
        snapshot_ids=pd.concat(snapshot_parts, ignore_index=True),
    )


def _resolve_step(clean_df: pd.DataFrame, total_requested: int, step_months: int | None) -> int:
    if step_months is not None:
        return step_months
    step = recommend_step_months(clean_df, total_requested)
    print(
        f"Auto-selected snapshot step: {step} month(s) apart "
        f"(feature window overlap: {max(0, FEATURE_PERIOD_MONTHS - step)} "
        f"of {FEATURE_PERIOD_MONTHS} months)."
    )
    return step


def build_train_val_test(
    clean_transactions_df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    step_months: int | None = None,
):
    """Development-compatible fixed split beginning at snapshot 1."""
    total = n_train + n_val + n_test
    step = _resolve_step(clean_transactions_df, total, step_months)
    available = max_available_snapshots(clean_transactions_df, step)
    if total > available:
        raise ValueError(f"Requested {total} snapshots but only {available} are fully labeled.")

    train_ids = list(range(1, n_train + 1))
    val_ids = list(range(n_train + 1, n_train + n_val + 1))
    test_ids = list(range(n_train + n_val + 1, total + 1))
    train = build_split(clean_transactions_df, train_ids, step)
    val = build_split(clean_transactions_df, val_ids, step)
    test = build_split(clean_transactions_df, test_ids, step)
    return (
        train.X, train.y, train.customer_ids,
        val.X, val.y, val.customer_ids,
        test.X, test.y, test.customer_ids,
    )


def rolling_snapshot_numbers(latest_snapshot: int, n_train: int, n_val: int, n_test: int) -> tuple[list[int], list[int], list[int]]:
    """Return a rolling window ending at ``latest_snapshot``.

    Example with 3/2/1 and latest=7: train=[2,3,4], val=[5,6], test=[7].
    """
    total = n_train + n_val + n_test
    first = latest_snapshot - total + 1
    if first < 1:
        raise ValueError(f"Need {total} snapshots but latest_snapshot={latest_snapshot}.")
    train = list(range(first, first + n_train))
    val = list(range(first + n_train, first + n_train + n_val))
    test = list(range(first + n_train + n_val, latest_snapshot + 1))
    return train, val, test


def build_rolling_train_val_test(
    clean_transactions_df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    step_months: int,
    data_through_date: pd.Timestamp | None = None,
    end_snapshot: int | None = None,
) -> DatasetBundle:
    """Build the newest fully-labeled rolling train/val/test window."""
    available = max_available_snapshots(
        clean_transactions_df, step_months, data_through_date=data_through_date
    )
    latest = available if end_snapshot is None else end_snapshot
    if latest > available:
        raise ValueError(f"end_snapshot={latest} is not fully labeled; latest available is {available}.")

    train_ids, val_ids, test_ids = rolling_snapshot_numbers(latest, n_train, n_val, n_test)
    print(f"Rolling split: train={train_ids} | val={val_ids} | test={test_ids}")
    return DatasetBundle(
        train=build_split(clean_transactions_df, train_ids, step_months),
        val=build_split(clean_transactions_df, val_ids, step_months),
        test=build_split(clean_transactions_df, test_ids, step_months),
        train_snapshots=train_ids,
        val_snapshots=val_ids,
        test_snapshots=test_ids,
    )
