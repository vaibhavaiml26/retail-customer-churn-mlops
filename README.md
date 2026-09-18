# Retail Customer Churn — Local Production Pipeline

This is the corrected local-production version of the uploaded project. It keeps the validated modelling design: a **5-month feature window**, **3-month churn outcome**, **3-month labeled-snapshot spacing**, monthly scoring, and XGBoost/Random Forest champion-challenger retraining.

## What was corrected

1. **Churn label:** only a future positive-quantity sale counts as renewed activity. A return alone does not make a customer active.
2. **Rolling retraining:** the split now moves forward, e.g. `1,2,3 | 4,5 | 6` then `2,3,4 | 5,6 | 7`.
3. **Snapshot readiness:** retraining checks the champion's `latest_snapshot_used`; it does not repeatedly retrain the same data.
4. **Champion comparison:** the deployed champion is evaluated frozen. It is never re-fitted just to compare against a challenger.
5. **XGBoost tuning:** `tune_xgboost` is implemented.
6. **Temporal CV:** hyperparameter CV uses forward-chaining snapshot folds when snapshot IDs are supplied.
7. **Retrain vs retune:** quarterly retraining uses the validated frozen parameters by default. Set `RETUNE_ON_RETRAIN=True` only when deliberately retuning.
8. **Curated layer:** cleaning happens before data enters the curated store.
9. **Idempotent ingestion:** SHA-256 + JSONL manifest prevents the exact monthly batch from being appended twice.
10. **Immutable raw archive:** every monthly source file is copied under `data/raw/YYYY-MM/` before curation.
11. **Explicit scoring cutoff:** monthly scoring requires `--snapshot-date`; it never infers the business cutoff from the last transaction timestamp.
12. **Delayed labels:** resolved outcomes are stored by `(CustomerID, snapshot_date)`.
13. **Monitoring:** feature PSI, prediction PSI, predicted churn rate, delayed ROC-AUC/PR-AUC/precision/recall/F1 are persisted.
14. **Registry lineage:** model version metadata contains feature schema, split snapshot IDs, environment versions, threshold, parameters, and latest labeled snapshot used.

## Folder structure

```text
retail_churn_production/
├── config.py
├── data_loader.py
├── data_validation.py
├── feature_engineering.py
├── dataset_builder.py
├── models.py
├── ingestion_manifest.py
├── pipeline_common.py
├── label_resolution.py
├── drift_monitoring.py
├── model_registry.py
├── score_monthly.py
├── retrain_quarterly.py
├── bootstrap_local.py
├── main.py
├── requirements.txt
├── tests/
├── data/
│   ├── raw/
│   ├── curated/
│   ├── manifests/
│   └── labels/
├── predictions/
├── monitoring/
└── models/registry/
```

## 1. Create the local environment

Windows example:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## 2. Bootstrap the historical model once

Place the two historical CSVs somewhere accessible, then:

```bash
python bootstrap_local.py --files online_retail_II_2009_10.csv online_retail_II_2010_11.csv
```

This creates the curated history and trains/registers the first champion using the newest fully resolved rolling snapshot window available in the historical data.

## 3. Score one completed month

If `transactions_2027_09.csv` contains **September 2027 only**, use an exclusive snapshot date of October 1:

```bash
python score_monthly.py --new-file transactions_2027_09.csv --snapshot-date 2027-10-01
```

The snapshot date means **data is complete up to, but not including, this date**. Features therefore use the five months ending at `2027-10-01`.

The monthly job:

```text
archive raw file
→ checksum/idempotency check
→ structural validation
→ clean
→ volume anomaly check
→ append curated history
→ build current features
→ load CURRENT champion
→ score official customers
→ score PREVIOUS champion silently as shadow (when available)
→ persist official + shadow predictions
→ behavioral + expected-temporal feature drift
→ prediction drift
→ resolve old official/shadow cohorts whose 3-month outcome just matured
→ calibration + delayed model metrics
→ persist monitoring/run logs
```

## 4. Run the retraining job

Schedule this monthly or quarterly. Running it too often is safe because it checks whether a genuinely new labeled snapshot exists:

```bash
python retrain_quarterly.py
```

For the historical production replay the configured `2/1/1` split rolls like this:

```text
latest=4: train [1,2] | val [3] | test [4]
latest=5: train [2,3] | val [4] | test [5]
latest=6: train [3,4] | val [5] | test [6]
```

For the richer model-development experiment, restore `3/2/1` in `config.py`.

The challenger is fit on the rolling training snapshots. Its threshold is selected on current validation data. The **frozen current champion** and challenger are then evaluated on the same validation set. Promotion uses ROC-AUC as the primary metric plus recall and Brier-score guardrails configured in `config.py`. The held-out test remains outside the promotion decision. When a challenger is promoted, the replaced champion becomes `PREVIOUS` and is available for shadow scoring.

## 5. Retraining is not retuning

Default:

```python
RETUNE_ON_RETRAIN = False
```

Quarterly candidates use the already validated model parameters. This is cheap and stable. To deliberately rerun GridSearchCV during a retrain cycle:

```python
RETUNE_ON_RETRAIN = True
```

Both RF and XGBoost tuning use **forward-chaining temporal CV** when snapshot IDs are supplied.

## 6. Local artifacts

- `data/raw/YYYY-MM/` — exact delivered source files.
- `data/manifests/ingested_files.jsonl` — checksum and ingestion audit trail.
- `data/curated/all_transactions.parquet` — cleaned, deduplicated history.
- `predictions/snapshot_YYYY-MM-DD__MODEL_VERSION.parquet` — official scored customer cohort.
- `predictions/shadow/` — previous-champion shadow predictions.
- `data/labels/customer_outcomes.parquet` — delayed ground truth.
- `monitoring/monitoring_history.csv` — immediate drift/prediction statistics.
- `monitoring/model_performance_history.csv` — delayed official/shadow production metrics.
- `monitoring/calibration_history.csv` — probability-decile calibration history.
- `monitoring/shadow_model_comparison_history.csv` — same-cohort official-vs-shadow comparison.
- `monitoring/run_log.jsonl` — pipeline execution audit.
- `models/registry/<model_type>/<version>/` — immutable model artifact + metadata + drift baselines.
- `models/registry/<model_type>/CURRENT` — current champion pointer.

## 7. Scheduling locally

For the MVP, use Windows Task Scheduler or cron:

- **Monthly:** `score_monthly.py` after the completed monthly transaction extract arrives.
- **Monthly:** `retrain_quarterly.py` after scoring. Its readiness check makes non-retrain months cheap no-ops.

The cloud phase can later replace local storage/scheduling/registry without changing the ML semantics.

## Important remaining local limitations

This is intentionally a local production-style MVP, not an imitation of a multinational bank's platform department.

- Alerts currently print to stderr; email/Teams/Slack is not wired.
- Storage is local Parquet/JSONL/CSV, not a transactional warehouse.
- There is no multi-process locking. Run one ingestion/retrain process at a time.
- No secrets are needed yet.
- No Docker/CI/CD is included yet; those are the next layer before cloud deployment.
