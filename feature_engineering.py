"""
Turns raw, cleaned transaction data into a per-customer feature table
(X) and churn label (y) for a single monthly "snapshot".

Vocabulary
----------
snapshot_no      : 1-indexed index of the snapshot (1st, 2nd, 3rd...).
step_months      : how many months apart consecutive snapshots start.
                    step_months >= FEATURE_PERIOD_MONTHS means feature
                    windows don't overlap at all; step_months <
                    FEATURE_PERIOD_MONTHS means they share that many
                    months of history (and likely many of the same
                    customers), which is a source of leakage between
                    splits, not just within cross-validation.
feature window   : the FEATURE_PERIOD_MONTHS of transactions used to
                    compute a customer's behaviour (RFM-style stats).
outcome window   : the OUTCOME_PERIOD_MONTHS right after the feature
                    window, used only to decide whether the customer
                    stayed active (label = 0) or went inactive (label = 1).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FEATURE_PERIOD_MONTHS, OUTCOME_PERIOD_MONTHS, SNAPSHOT_START


@dataclass(frozen=True)
class SnapshotWindow:
    feature_start: pd.Timestamp
    feature_end: pd.Timestamp
    outcome_start: pd.Timestamp
    outcome_end: pd.Timestamp


def summarize_purchase_intervals(clean_transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, per customer, the average number of days between their
    purchase transactions across the WHOLE cleaned dataset (not windowed
    to a snapshot), then report percentiles of that distribution.

    Use this before changing FEATURE_PERIOD_MONTHS / OUTCOME_PERIOD_MONTHS
    in config.py: it tells you the natural repurchase cadence of your
    customers, so window sizes can be picked to match real behaviour
    rather than purely to fit more snapshots into the calendar.

    - If the outcome window is much shorter than typical repurchase gaps,
      you'll mislabel a lot of genuinely-returning customers as churned.
    - If the feature window is too short to contain even 2 purchases for
      a typical customer, days-between-purchase features will be mostly
      NaN / noisy for them.
    """
    sales = clean_transactions_df[clean_transactions_df["Quantity"] > 0].copy()
    sorted_tx = sales.sort_values(["CustomerID", "InvoiceDate"])
    unique_tx = sorted_tx[["CustomerID", "Invoice", "InvoiceDate"]].drop_duplicates()
    unique_tx["days_between_tx"] = unique_tx.groupby("CustomerID")[
        "InvoiceDate"
    ].diff().dt.days

    avg_days_between_tx = unique_tx.groupby("CustomerID")["days_between_tx"].mean().dropna()

    percentiles = avg_days_between_tx.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    summary = pd.DataFrame(
        {
            "avg_days_between_purchases": percentiles.values,
        },
        index=[f"p{int(q*100)}" for q in percentiles.index],
    )

    print(
        f"{len(avg_days_between_tx)} customers with 2+ purchases "
        f"(out of {unique_tx['CustomerID'].nunique()} total customers)."
    )
    print(summary)
    print(
        f"Median repurchase gap: {avg_days_between_tx.median():.1f} days "
        f"(~{avg_days_between_tx.median()/30:.1f} months)"
    )
    return summary


def get_snapshot_window(snapshot_no: int, step_months: int) -> SnapshotWindow:
    """
    Compute the four date boundaries for a given snapshot number, spaced
    `step_months` apart. Snapshot 1 always starts at SNAPSHOT_START.
    """
    feature_start = SNAPSHOT_START + pd.DateOffset(
        months=(snapshot_no - 1) * step_months
    )
    feature_end = feature_start + pd.DateOffset(months=FEATURE_PERIOD_MONTHS)
    outcome_start = feature_end
    outcome_end = outcome_start + pd.DateOffset(months=OUTCOME_PERIOD_MONTHS)
    return SnapshotWindow(feature_start, feature_end, outcome_start, outcome_end)


def max_available_snapshots(
    df: pd.DataFrame, step_months: int, data_through_date: pd.Timestamp | None = None
) -> int:
    """Return the largest snapshot whose *entire* outcome window is known.

    In production, ``data_through_date`` should be the exclusive boundary of
    the latest complete monthly batch (for example 2027-10-01 means data is
    complete through 2027-09-30). Falling back to the largest event timestamp
    preserves compatibility with the historical project data.
    """
    coverage_end = pd.Timestamp(data_through_date) if data_through_date is not None else df["InvoiceDate"].max()
    snapshot_no = 1
    while get_snapshot_window(snapshot_no, step_months).outcome_end <= coverage_end:
        snapshot_no += 1
    return snapshot_no - 1


def recommend_step_months(
    df: pd.DataFrame, total_snapshots: int, max_step: int = 12,
    data_through_date: pd.Timestamp | None = None,
) -> int:
    """
    Find the largest whole-month spacing that still fits `total_snapshots`
    within the available data. Larger spacing means less overlap between
    consecutive snapshots' feature windows (and less leakage of the same
    customer across splits), so we want the largest step that still fits.
    """
    best_step = 1
    for step in range(1, max_step + 1):
        if max_available_snapshots(df, step, data_through_date=data_through_date) >= total_snapshots:
            best_step = step
        else:
            break  # feasibility only shrinks as step grows, so stop early
    return best_step


def split_feature_outcome(
    df: pd.DataFrame, snapshot_no: int, step_months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED: kept only in case external code imports it. The feature/
    outcome slicing logic now lives inline in generate_snapshot_dataset
    and _build_feature_slice, since build_customer_features was
    refactored to take a window directly (needed to share code with the
    label-free generate_scoring_dataset).
    """
    window = get_snapshot_window(snapshot_no, step_months)
    feature_df = df[
        (df["InvoiceDate"] >= window.feature_start) & (df["InvoiceDate"] < window.feature_end)
    ].copy()
    outcome_df = df[
        (df["InvoiceDate"] >= window.outcome_start) & (df["InvoiceDate"] < window.outcome_end)
    ].copy()
    return feature_df, outcome_df


def _compute_rfm_stats(
    transactions: pd.DataFrame, window: "SnapshotWindow"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute per-customer RFM-style aggregates for one side of the ledger
    (either sales-only or returns-only rows should be passed in).

    Takes the window directly (rather than snapshot_no/step_months) so the
    same logic serves both labeled-snapshot generation (generate_snapshot_dataset)
    and label-free scoring (generate_scoring_dataset) -- scoring has no
    "snapshot_no" in the training sequence, just an as-of date.

    Returns (revenue, item_count, tx_count, avg_days_between_tx, recency).
    """
    revenue = (transactions["Price"] * transactions["Quantity"].abs()).groupby(
        transactions["CustomerID"]
    ).sum()

    item_count = transactions.groupby("CustomerID")["Quantity"].sum().abs()

    tx_count = transactions.groupby("CustomerID")["Invoice"].nunique()

    sorted_tx = transactions.sort_values(["CustomerID", "InvoiceDate"])
    unique_tx = sorted_tx[["CustomerID", "Invoice", "InvoiceDate"]].drop_duplicates()
    unique_tx["days_between_tx"] = unique_tx.groupby("CustomerID")[
        "InvoiceDate"
    ].diff().dt.days
    avg_days_between_tx = unique_tx.groupby("CustomerID")["days_between_tx"].mean()

    last_tx_date = unique_tx.groupby("CustomerID")["InvoiceDate"].max()
    recency = (window.feature_end - last_tx_date.dt.normalize()).dt.days

    return revenue, item_count, tx_count, avg_days_between_tx, recency


def compute_tenure_days(
    full_clean_df: pd.DataFrame, customers: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.Series:
    """
    Days between each customer's first-ever purchase and `as_of_date`
    (normally the feature window's end), using ALL history up to that
    date -- not just the feature window.

    This intentionally looks further back than the feature window: it's
    still legitimate (not leakage) because it only uses data strictly
    before `as_of_date`, the same cutoff the rest of the features respect,
    and it captures true customer age rather than an age clipped to
    whatever the window happens to contain.
    """
    sales_to_date = full_clean_df[
        (full_clean_df["Quantity"] > 0) & (full_clean_df["InvoiceDate"] < as_of_date)
    ]
    first_purchase = sales_to_date.groupby("CustomerID")["InvoiceDate"].min()
    tenure_days = (as_of_date - first_purchase).dt.days
    return customers["CustomerID"].map(tenure_days)


def compute_momentum_features(sales: pd.DataFrame, window: SnapshotWindow) -> pd.DataFrame:
    """
    Split the feature window into an earlier half and a more recent half,
    and compute each customer's revenue/transaction count in each half.

    Returns a dataframe indexed by CustomerID with:
      recent_revenue, revenue_momentum   -- (recent - early) / (recent + early),
                                             bounded in [-1, 1]; positive means
                                             spending is accelerating, negative
                                             means it's tailing off. 0 when a
                                             customer had no sales in either half.
      recent_tx_count, tx_count_momentum -- same idea, for transaction counts.

    These catch customers whose behaviour is changing within the window --
    a static window-total treats "steady" and "was active, went quiet"
    customers identically if their totals happen to match.
    """
    midpoint = window.feature_start + (window.feature_end - window.feature_start) / 2

    early = sales[sales["InvoiceDate"] < midpoint]
    recent = sales[sales["InvoiceDate"] >= midpoint]

    early_revenue = (early["Price"] * early["Quantity"]).groupby(early["CustomerID"]).sum()
    recent_revenue = (recent["Price"] * recent["Quantity"]).groupby(recent["CustomerID"]).sum()
    early_tx = early.groupby("CustomerID")["Invoice"].nunique()
    recent_tx = recent.groupby("CustomerID")["Invoice"].nunique()

    result = pd.DataFrame(
        {
            "early_revenue": early_revenue,
            "recent_revenue": recent_revenue,
            "early_tx_count": early_tx,
            "recent_tx_count": recent_tx,
        }
    ).fillna(0)

    revenue_denom = result["early_revenue"] + result["recent_revenue"]
    result["revenue_momentum"] = np.where(
        revenue_denom > 0,
        (result["recent_revenue"] - result["early_revenue"]) / revenue_denom,
        0.0,
    )

    tx_denom = result["early_tx_count"] + result["recent_tx_count"]
    result["tx_count_momentum"] = np.where(
        tx_denom > 0,
        (result["recent_tx_count"] - result["early_tx_count"]) / tx_denom,
        0.0,
    )

    return result[["recent_revenue", "revenue_momentum", "recent_tx_count", "tx_count_momentum"]]


def build_customer_features(
    customers: pd.DataFrame,
    sales: pd.DataFrame,
    returns: pd.DataFrame,
    window: SnapshotWindow,
    full_clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach sale-side, return-side, tenure, and momentum features to each
    customer, as of `window.feature_end`. Takes the window directly so
    this same function serves both training-snapshot generation and
    label-free scoring (see generate_snapshot_dataset vs.
    generate_scoring_dataset).
    """
    customer_features = customers.copy()

    revenue, items, tx_count, avg_days, recency = _compute_rfm_stats(sales, window)
    customer_features["total_sales_revenue"] = customer_features["CustomerID"].map(revenue)
    customer_features["total_items_purchased"] = customer_features["CustomerID"].map(items)
    customer_features["total_sale_txs"] = customer_features["CustomerID"].map(tx_count)
    customer_features["avg_sale_days"] = customer_features["CustomerID"].map(avg_days)
    customer_features["recency"] = customer_features["CustomerID"].map(recency)

    r_revenue, r_items, r_tx_count, r_avg_days, _ = _compute_rfm_stats(returns, window)
    customer_features["total_return_amount"] = (
        customer_features["CustomerID"].map(r_revenue).fillna(0)
    )
    customer_features["total_items_returned"] = (
        customer_features["CustomerID"].map(r_items).fillna(0)
    )
    customer_features["total_return_txs"] = (
        customer_features["CustomerID"].map(r_tx_count).fillna(0)
    )
    customer_features["avg_return_days"] = (
        customer_features["CustomerID"].map(r_avg_days).fillna(0)
    )

    customer_features["tenure_days"] = compute_tenure_days(
        full_clean_df, customers, window.feature_end
    )

    momentum = compute_momentum_features(sales, window)
    for col in momentum.columns:
        customer_features[col] = customer_features["CustomerID"].map(momentum[col]).fillna(0)

    return customer_features


def _build_feature_slice(
    clean_transactions_df: pd.DataFrame, window: SnapshotWindow
) -> pd.DataFrame:
    """Shared by both dataset builders: slice + compute features for a window."""
    feature_df = clean_transactions_df[
        (clean_transactions_df["InvoiceDate"] >= window.feature_start)
        & (clean_transactions_df["InvoiceDate"] < window.feature_end)
    ].copy()

    customers = feature_df[["CustomerID"]].drop_duplicates()
    sales = feature_df[feature_df["Quantity"] > 0].copy()
    returns = feature_df[feature_df["Quantity"] < 0].copy()

    customer_features = build_customer_features(
        customers, sales, returns, window, clean_transactions_df
    )

    fill_zero_cols = ["total_sales_revenue", "total_items_purchased", "total_sale_txs"]
    customer_features[fill_zero_cols] = customer_features[fill_zero_cols].fillna(0)
    return customer_features


def generate_snapshot_dataset(
    clean_transactions_df: pd.DataFrame, snapshot_no: int, step_months: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build (X, y, groups) for one LABELED training/eval snapshot:
      - X: one row per customer active in the feature window, with RFM,
           tenure, and momentum features computed as of the window's end.
      - y: 1 if the customer made NO positive-quantity SALE in the outcome
           window ("inactive_90d" / churned), else 0. Returns alone do not
           count as renewed activity. Requires the
           outcome window to have already fully elapsed in the data --
           see max_available_snapshots / recommend_step_months. This is
           why labeled snapshots are only available on the
           SNAPSHOT_STEP_MONTHS cadence (e.g. quarterly), not every time
           new data arrives -- the label needs OUTCOME_PERIOD_MONTHS of
           future data to be knowable at all. For scoring currently-active
           customers with no wait, see generate_scoring_dataset instead.
      - groups: the CustomerID for each row, tagged separately from X
           (never used as a model feature) so callers can do
           group-aware cross-validation.
    """
    window = get_snapshot_window(snapshot_no, step_months)
    print(
        f"Snapshot {snapshot_no} (step={step_months}mo): feature "
        f"[{window.feature_start.date()} -> {window.feature_end.date()}), "
        f"outcome [{window.outcome_start.date()} -> {window.outcome_end.date()})"
    )

    customer_features = _build_feature_slice(clean_transactions_df, window)

    outcome_df = clean_transactions_df[
        (clean_transactions_df["InvoiceDate"] >= window.outcome_start)
        & (clean_transactions_df["InvoiceDate"] < window.outcome_end)
    ]
    # Churn is defined by absence of a future SALE. A return-only customer
    # remains inactive for this target; otherwise returns would falsely turn
    # churners into active customers.
    future_sales = outcome_df[outcome_df["Quantity"] > 0]
    active_in_outcome = future_sales["CustomerID"].drop_duplicates()
    customer_features["inactive_90d"] = (
        ~customer_features["CustomerID"].isin(active_in_outcome)
    ).astype(int)

    X = customer_features.drop(columns=["CustomerID", "inactive_90d"])
    y = customer_features["inactive_90d"]
    groups = customer_features["CustomerID"]
    return X, y, groups


def generate_scoring_dataset(
    clean_transactions_df: pd.DataFrame, as_of_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, groups) for customers active in the FEATURE_PERIOD_MONTHS
    trailing `as_of_date` -- NO outcome/label, because scoring doesn't
    need one: you're asking "what's this customer's churn risk right
    now", not building a labeled training example.

    Unlike generate_snapshot_dataset, this can be run every time new
    data arrives (e.g. every month) even though labeled snapshots (for
    retraining) are only available on the SNAPSHOT_STEP_MONTHS cadence --
    scoring never has to wait for an outcome window to resolve, because
    it isn't computing one.
    """
    window = SnapshotWindow(
        feature_start=as_of_date - pd.DateOffset(months=FEATURE_PERIOD_MONTHS),
        feature_end=as_of_date,
        outcome_start=as_of_date,  # unused for scoring; kept for a consistent window shape
        outcome_end=as_of_date,
    )
    print(
        f"Scoring as of {as_of_date.date()}: feature "
        f"[{window.feature_start.date()} -> {window.feature_end.date()})"
    )

    customer_features = _build_feature_slice(clean_transactions_df, window)

    X = customer_features.drop(columns=["CustomerID"])
    groups = customer_features["CustomerID"]
    return X, groups
