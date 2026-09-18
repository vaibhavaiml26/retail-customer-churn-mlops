"""Hard structural validation plus soft/hard monthly-volume anomaly checks."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID"}


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        lines.extend(f"  ERROR: {x}" for x in self.errors)
        lines.extend(f"  WARNING: {x}" for x in self.warnings)
        return "\n".join(lines)


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    errors = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return ValidationResult(False, [f"Missing required columns: {sorted(missing)}"])
    if df.empty:
        errors.append("File has zero rows.")
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        errors.append("InvoiceDate is not parsed as datetime.")
    for col in ["Quantity", "Price"]:
        try:
            pd.to_numeric(df[col])
        except (ValueError, TypeError) as exc:
            errors.append(f"{col} could not be parsed as numeric: {exc}")
    null_customer_frac = float(df["CustomerID"].isna().mean())
    if null_customer_frac > 0.95:
        errors.append(f"{null_customer_frac:.0%} of rows have no CustomerID.")
    dup_rate = float(
        df.duplicated(subset=["Invoice", "StockCode", "Quantity", "InvoiceDate"]).mean()
    )
    if dup_rate > 0.20:
        errors.append(f"{dup_rate:.0%} of rows are duplicate transaction lines.")
    return ValidationResult(not errors, errors=errors)


def compute_monthly_baseline(historical_clean_df: pd.DataFrame, trailing_months: int = 12) -> pd.DataFrame:
    if historical_clean_df.empty:
        return pd.DataFrame(index=["tx_count", "customer_count", "revenue"], columns=["mean", "std"])
    df = historical_clean_df.copy()
    df["month"] = df["InvoiceDate"].dt.to_period("M")
    df["line_revenue"] = df["Price"] * df["Quantity"]
    monthly = df.groupby("month").agg(
        tx_count=("Invoice", "nunique"),
        customer_count=("CustomerID", "nunique"),
        revenue=("line_revenue", "sum"),
    )
    recent = monthly.tail(trailing_months)
    return pd.DataFrame({
        "mean": recent.mean(),
        "std": recent.std(ddof=0).replace(0, np.nan),
    })


def detect_volume_anomalies(
    new_month_df: pd.DataFrame,
    baseline: pd.DataFrame,
    soft_z_threshold: float = 3.0,
    hard_z_threshold: float = 5.0,
) -> ValidationResult:
    observed = {
        "tx_count": new_month_df["Invoice"].nunique(),
        "customer_count": new_month_df["CustomerID"].nunique(),
        "revenue": (new_month_df["Price"] * new_month_df["Quantity"]).sum(),
    }
    errors, warnings = [], []
    for metric, value in observed.items():
        if metric not in baseline.index:
            continue
        mean = baseline.loc[metric, "mean"]
        std = baseline.loc[metric, "std"]
        if pd.isna(mean) or pd.isna(std):
            continue
        z = (value - mean) / std
        msg = f"{metric}={value:,.0f} is {z:+.1f} SD from trailing mean={mean:,.0f}."
        if abs(z) > hard_z_threshold:
            errors.append(msg + " Exceeds hard threshold.")
        elif abs(z) > soft_z_threshold:
            warnings.append(msg + " Exceeds soft threshold.")
    return ValidationResult(not errors, errors=errors, warnings=warnings)
