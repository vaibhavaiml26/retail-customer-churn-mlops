# Production hardening update: items 1–5

Replace the listed Python files in your existing project. **Do not delete your
current curated data, registry, predictions, labels, or monitoring history.**
The update is backward-compatible with the replay state you already produced.

## Files to replace

- `config.py`
- `pipeline_common.py`
- `model_registry.py`
- `models.py`
- `drift_monitoring.py`
- `score_monthly.py`
- `label_resolution.py`
- `retrain_quarterly.py`

Optional utility:

- `backfill_shadow_predictions.py`

Tests added/updated under `tests/` are included in the ZIP.

## 1. Previous-champion shadow scoring

`model_registry.py` now maintains `CURRENT` and `PREVIOUS` pointers. Existing
registries that do not yet have a `PREVIOUS` file automatically infer the most
recent registered model other than `CURRENT`, so your current replay registry
does not need to be rebuilt.

Every new monthly score writes the official prediction as before and, when a
previous champion exists, silently writes a second prediction set under:

`predictions/shadow/`

Shadow predictions never drive the business classification. They exist only
for later apples-to-apples evaluation on the same customer cohort and outcome
period.

For your existing replay, you can immediately backfill the September cohort:

```powershell
python .\backfill_shadow_predictions.py --snapshot-date 2011-09-01
```

Because your curated history already reaches 2011-12-01, that shadow cohort is
mature and can be evaluated immediately. The official-vs-shadow comparison is
written to:

`monitoring/shadow_model_comparison_history.csv`

## 2. `tenure_days` monitoring

`tenure_days` remains a model feature, but is classified as
`expected_temporal` in the feature drift report. It no longer causes the
behavioral drift alert by itself.

Monthly monitoring now records both:

- `max_behavioral_feature_psi`
- `temporal_psi_tenure_days`

The full feature drift CSV also contains a `drift_type` column.

## 3. Calibration monitoring

Delayed evaluation now adds:

- Brier score
- mean churn probability
- actual churn rate
- calibration gap
- expected calibration error (ECE)
- predicted churn rate
- mean threshold used

A decile calibration history is written to:

`monitoring/calibration_history.csv`

## 4. Model-performance history

`monitoring/model_performance_history.csv` now stores one row per resolved
`(snapshot_date, model_version, prediction_role)` prediction set. Existing old
rows are preserved and treated as `official` when the old file has no
`prediction_role` column.

The CSV history writer now safely evolves its schema, so adding the new columns
will not corrupt your existing monitoring history.

## 5. Promotion guardrails

Promotion still uses ROC-AUC as the primary metric, but a challenger is blocked
if any configured guardrail is violated:

- ROC-AUC regression > 0.005
- recall regression > 0.05
- Brier score degradation > 0.02

The thresholds are in `config.py` and can be changed deliberately later.

## Validation

The updated project passes the local unit suite. One Parquet integration test
may skip in an environment where `pyarrow` is not installed; `pyarrow` remains
in `requirements.txt` for the actual project environment.
