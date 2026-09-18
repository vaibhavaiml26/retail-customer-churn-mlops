import pandas as pd
from label_resolution import build_actual_outcomes


def test_delayed_labels_use_future_sales_not_returns(small_transactions):
    labels = build_actual_outcomes(
        small_transactions,
        pd.Series([1.0, 2.0]),
        pd.Timestamp("2010-05-01"),
    )
    by_customer = labels.set_index("CustomerID")["inactive_90d"].to_dict()
    assert by_customer[1.0] == 1
    assert by_customer[2.0] == 0
