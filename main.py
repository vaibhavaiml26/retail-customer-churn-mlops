"""Development/research entry point using the same corrected feature/model code."""
import config
from data_loader import clean_transactions, load_transactions
from dataset_builder import build_rolling_train_val_test
from models import (
    build_production_candidate,
    evaluate_fitted_model,
    find_best_threshold,
    get_feature_importance,
    tune_random_forest,
    tune_xgboost,
)


def main():
    clean = clean_transactions(load_transactions(config.DATA_FILES))
    bundle = build_rolling_train_val_test(
        clean,
        config.N_TRAIN_SNAPSHOTS,
        config.N_VAL_SNAPSHOTS,
        config.N_TEST_SNAPSHOTS,
        config.SNAPSHOT_STEP_MONTHS,
    )
    tuner = {"random_forest": tune_random_forest, "xgboost": tune_xgboost}[config.MODEL_TYPE]
    if config.RUN_HYPERPARAMETER_SEARCH:
        model, result, _ = tuner(
            bundle.train.X, bundle.train.y, bundle.val.X, bundle.val.y,
            groups_train=bundle.train.customer_ids,
            snapshot_ids_train=bundle.train.snapshot_ids,
        )
        print("Best params:", result["params"])
        print("Temporal CV ROC-AUC:", result["cv_roc_auc"])
    else:
        model, params = build_production_candidate(config.MODEL_TYPE)
        model.fit(bundle.train.X, bundle.train.y)
        print("Frozen params:", params)

    threshold, table = find_best_threshold(model, bundle.val.X, bundle.val.y, config.THRESHOLD_METRIC)
    print(table)
    print(get_feature_importance(model, bundle.train.X.columns))
    print(evaluate_fitted_model(model, bundle.test.X, bundle.test.y, "Held-out test", threshold))


if __name__ == "__main__":
    main()
