import numpy as np
import pandas as pd
from models import evaluate_fitted_model, make_forward_chaining_cv


class PredictOnlyModel:
    def fit(self, *args, **kwargs):
        raise AssertionError("evaluate_fitted_model must not call fit")

    def predict_proba(self, X):
        p = np.asarray(X["p"], dtype=float)
        return np.column_stack([1 - p, p])


def test_evaluate_fitted_model_does_not_refit():
    model = PredictOnlyModel()
    X = pd.DataFrame({"p": [0.1, 0.9, 0.2, 0.8]})
    y = pd.Series([0, 1, 0, 1])
    metrics = evaluate_fitted_model(model, X, y, threshold=0.5)
    assert metrics["roc_auc"] == 1.0


def test_temporal_cv_is_forward_only():
    snapshot_ids = pd.Series([2, 2, 3, 3, 4, 4])
    folds = make_forward_chaining_cv(snapshot_ids)
    assert len(folds) == 2
    tr0, va0 = folds[0]
    assert set(snapshot_ids.iloc[tr0]) == {2}
    assert set(snapshot_ids.iloc[va0]) == {3}
    tr1, va1 = folds[1]
    assert set(snapshot_ids.iloc[tr1]) == {2, 3}
    assert set(snapshot_ids.iloc[va1]) == {4}
