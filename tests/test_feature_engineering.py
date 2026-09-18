import pandas as pd
from feature_engineering import generate_scoring_dataset, generate_snapshot_dataset


def test_return_only_outcome_counts_as_churn(small_transactions):
    X, y, customers = generate_snapshot_dataset(small_transactions, snapshot_no=1, step_months=3)
    labels = dict(zip(customers, y))
    assert labels[1.0] == 1
    assert labels[2.0] == 0


def test_training_scoring_feature_parity(small_transactions):
    X_train, _, customers_train = generate_snapshot_dataset(small_transactions, 1, 3)
    X_score, customers_score = generate_scoring_dataset(small_transactions, pd.Timestamp("2010-05-01"))
    assert list(X_train.columns) == list(X_score.columns)

    train = X_train.copy(); train["CustomerID"] = customers_train.values
    score = X_score.copy(); score["CustomerID"] = customers_score.values
    train = train.sort_values("CustomerID").reset_index(drop=True)
    score = score.sort_values("CustomerID").reset_index(drop=True)
    pd.testing.assert_frame_equal(train, score, check_dtype=False)
