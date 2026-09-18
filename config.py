"""Central configuration for the local production churn pipeline."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
CURATED_DATA_PATH = DATA_ROOT / "curated" / "all_transactions.parquet"
INGESTION_MANIFEST_PATH = DATA_ROOT / "manifests" / "ingested_files.jsonl"
LABELS_PATH = DATA_ROOT / "labels" / "customer_outcomes.parquet"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
SHADOW_PREDICTIONS_DIR = PREDICTIONS_DIR / "shadow"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
REGISTRY_ROOT = PROJECT_ROOT / "models" / "registry"

# Historical bootstrap inputs. Override with CLI args if the files live elsewhere.
DATA_FILES = ["online_retail_II_2009_10.csv", "online_retail_II_2010_11.csv"]
DATE_FORMAT = "%d-%m-%Y %H:%M"

# Snapshot definition validated in the project.
SNAPSHOT_START = pd.Timestamp("2009-12-01")
FEATURE_PERIOD_MONTHS = 5
OUTCOME_PERIOD_MONTHS = 3
SNAPSHOT_STEP_MONTHS = 3

# Historical production replay uses 2/1/1 so enough future months remain to
# exercise monthly scoring, delayed labels, and rolling retraining. For the
# richer development experiment you can restore 3/2/1.
N_TRAIN_SNAPSHOTS = 2
N_VAL_SNAPSHOTS = 1
N_TEST_SNAPSHOTS = 1

RANDOM_STATE = 42
MODEL_TYPE = "xgboost"  # "xgboost" or "random_forest"
THRESHOLD_METRIC = "youden_j"
RUN_HYPERPARAMETER_SEARCH = True  # development main.py only

RF_PARAM_GRID = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [4, 6, 8, 10],
    "classifier__min_samples_split": [10, 20, 50],
    "classifier__min_samples_leaf": [5, 10, 20],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__max_samples": [0.6, 0.8, None],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

BEST_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_split": 20,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "max_samples": 0.8,
}
BEST_RF_THRESHOLD = 0.40

BEST_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
}
BEST_XGB_THRESHOLD = 0.35

# Production policy: quarterly retraining does not imply quarterly retuning.
RETUNE_ON_RETRAIN = False

# Champion/challenger guardrails. ROC-AUC is the primary criterion, but a
# candidate is also blocked if recall or Brier score deteriorates too much.
PROMOTION_METRIC = "roc_auc"
PROMOTION_MAX_REGRESSION = 0.005
PROMOTION_MAX_RECALL_REGRESSION = 0.05
PROMOTION_MAX_BRIER_REGRESSION = 0.02

# Drift policy. tenure_days is monotonic with calendar time, so report it
# separately instead of allowing it to dominate the behavioral drift alert.
DRIFT_PSI_WARNING = 0.10
DRIFT_PSI_HIGH = 0.25
DRIFT_BASELINE_SAMPLE_SIZE = 2000
EXPECTED_TEMPORAL_DRIFT_FEATURES = ["tenure_days"]

# Keep the immediately previous champion scoring silently alongside CURRENT.
SHADOW_SCORING_ENABLED = True

# Delayed calibration monitoring.
CALIBRATION_BINS = 10

RISK_BINS = [0.0, 0.30, 0.60, 1.0]
RISK_LABELS = ["low", "medium", "high"]
