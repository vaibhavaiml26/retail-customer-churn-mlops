"""Raw-data loading and deterministic customer-level cleaning."""
from pathlib import Path
import pandas as pd
import config


def load_transactions(filepaths) -> pd.DataFrame:
    """Read one or more CSV files, parse InvoiceDate, concatenate and sort."""
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]

    frames = []
    for path in filepaths:
        df = pd.read_csv(path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format=config.DATE_FORMAT)
        frames.append(df)

    if not frames:
        raise ValueError("No transaction files were supplied.")

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows attributable to a customer and normalize ordering.

    The feature logic intentionally retains both positive sales and negative
    return rows. Rows without CustomerID cannot contribute to a customer-level
    churn model and are removed here.
    """
    clean = df.dropna(subset=["CustomerID"]).copy()
    clean = clean.drop_duplicates()
    return clean.sort_values("InvoiceDate").reset_index(drop=True)
