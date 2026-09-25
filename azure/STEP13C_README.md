# Step 13C - Monitoring Alerts

This package adds a fourth Azure ML pipeline component named `monitoring_alerts`.

Before delayed outcomes mature:
`monthly_score -> monitoring_alerts`

When delayed outcomes are available:
`monthly_score + delayed_evaluation -> monitoring_alerts`

`retrain_check` remains independent and only appears on the quarterly cadence.

Copy these into your project's `azure/` folder:
- `evaluate_monitoring_alerts.py`
- `submit_pipeline_job.py`

Preserve the working values from your current local submitter for workspace, data assets, compute, environment, and AS_OF_DATE. In particular, do not replace a working BOOTSTRAP_DATA or REPLAY_DATA name with an older placeholder from this package.

Alerts are persisted under:
`azureml://datastores/workspaceblobstore/paths/retail-churn/alert-history/<AS_OF_DATE>`

Each run writes:
- `alert_summary.json`
- `alert_record.csv`
- `alert_signals.csv`

Immediate alerts:
- prediction PSI: warning >= 0.10, critical >= 0.25
- max behavioral feature PSI excluding `tenure_days`: warning >= 0.10, critical >= 0.25

Delayed-performance warning thresholds:
- ROC-AUC drop > 0.005
- recall drop > 0.05
- Brier worsening > 0.02

Critical performance regression defaults to 2x the warning threshold.

Reference model metrics are selected in this order:
1. held-out `test_metrics`
2. `validation_metrics`
3. `metrics`

Warnings and critical alerts are deliberately non-blocking: they are persisted as monitoring state and do not fail otherwise successful scoring/retraining runs.

If delayed prediction `model_lineage` differs from the model supplied to the alert component, delayed performance is marked `NOT_COMPARABLE` instead of comparing against the wrong model version.
