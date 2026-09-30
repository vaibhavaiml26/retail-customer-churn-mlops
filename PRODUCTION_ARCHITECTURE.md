# Production Architecture

## Overview

The production architecture transforms raw grocery-retail transaction data into customer-level churn predictions, persists those predictions for delayed evaluation, monitors model health, and performs controlled champion-challenger retraining.

The complete lifecycle is:

```text
                         GitHub Repository
                               |
                        CI: Compile + Test
                               |
                         Manual CD Trigger
                               |
                         OIDC / Entra ID
                               |
                          Azure IAM/RBAC
                               |
                               v
+---------------------------------------------------------------+
|                     AZURE ML PIPELINE                         |
|                                                               |
|  Raw Transaction Data                                        |
|          |                                                    |
|          v                                                    |
|  Immutable Audit Layer                                       |
|          |                                                    |
|          v                                                    |
|  Data Validation                                             |
|          |                                                    |
|          v                                                    |
|  Data Cleaning                                               |
|          |                                                    |
|          v                                                    |
|  Temporal Leakage Checks                                     |
|          |                                                    |
|          v                                                    |
|  Snapshot Construction                                       |
|          |                                                    |
|          v                                                    |
|  14 Customer Features + Future Outcome                       |
|          |                                                    |
|          v                                                    |
|  Train / Validation / Test                                   |
|          |                                                    |
|          v                                                    |
|  Model Training & Tuning                                     |
|          |                                                    |
|          v                                                    |
|  Model Registry                                              |
|          |                                                    |
|          v                                                    |
|  Champion Model                                              |
|          |                                                    |
|          v                                                    |
|  Monthly Batch Scoring                                       |
|          |                                                    |
|          +------------> Prediction History                    |
|          |                                                    |
|          v                                                    |
|  Immediate Drift Monitoring                                  |
|          |                                                    |
|          v                                                    |
|  Outcome Matures After 3 Months                              |
|          |                                                    |
|          v                                                    |
|  Delayed Ground-Truth Evaluation                             |
|          |                                                    |
|          +------------> Performance History                   |
|          |                                                    |
|          v                                                    |
|  Monitoring & Alerts                                         |
|          |                                                    |
|          +------------> Alert History                         |
|          |                                                    |
|          v                                                    |
|  Quarterly Retraining Readiness                              |
|          |                                                    |
|          v                                                    |
|  Champion vs Challenger                                      |
|          |                                                    |
|          v                                                    |
|  ROC-AUC / Recall / Brier Guardrails                         |
|          |                                                    |
|     +----+-----+                                              |
|     v          v                                              |
|  Promote     Retain                                           |
| Challenger   Champion                                         |
+---------------------------------------------------------------+
```

---

## 1. Raw Transaction Data and Audit Layer

Almost two years of grocery-retail transaction data was retained in an immutable raw-data layer before any cleaning or modelling transformations were applied.

The raw layer provides:

- Auditability
- Reproducibility
- Reprocessing capability
- Separation between source data and derived ML data

```text
Source Transaction Data
        |
        v
Immutable Raw Data
        |
        +-- Audit trail
        +-- Reproducibility
        +-- Reprocessing
```

---

## 2. Data Validation

Validation is performed before downstream processing to detect malformed, duplicated, missing, or temporally invalid transactions.

Checks include:

- Transaction-date validation
- Duplicate detection
- Required-field validation
- Customer identifier checks
- Quantity and price consistency
- Transaction-window validation
- Basic distribution/data-quality checks

Feature drift is evaluated later, once customer-level features have been generated.

---

## 3. Data Cleaning

Cleaning removes records that should not represent customer purchasing behaviour, including irrelevant administrative or supply-related transactions.

The logic also distinguishes:

```text
Positive Quantity -> Sale
Negative Quantity -> Return / Cancellation
```

Rows without a usable customer identifier are excluded because the modelling problem is customer-level.

---

## 4. Temporal Leakage Checks

The system enforces a strict temporal boundary between feature calculation and outcome generation.

```text
<--------- Feature Window --------->|<------ Outcome Window ------>
                                     T
Historical transactions             Future transactions
```

Rules include:

- Feature calculations use only transactions within the historical feature window.
- Outcome labels use only transactions from the future outcome window.
- Future transactions cannot influence training features.
- Older or out-of-window transactions cannot incorrectly influence the target period.

---

## 5. Snapshot Construction

Each modelling snapshot consists of:

```text
5-month feature window
        +
3-month outcome window
```

Historical snapshots are spaced three months apart, producing a two-month overlap between adjacent five-month feature windows.

This balances reduced temporal correlation with the need to retain enough training data from a limited source history.

---

## 6. Feature Engineering

Validated and cleaned transactions are converted into approximately 14 customer-level behavioural features covering:

- Revenue
- Items purchased
- Sale transactions
- Purchase frequency
- Recency
- Return behaviour
- Recent revenue
- Customer tenure
- Recent transaction count
- Revenue momentum
- Transaction momentum

Each feature record is identified by:

```text
(CustomerID, SnapshotDate)
```

---

## 7. Outcome Generation

The churn outcome is based on the following three months of behaviour.

```text
At least one valid positive sale -> Active -> inactive_90d = 0
No valid positive sale           -> Churn  -> inactive_90d = 1
```

Returns alone do not count as activity.

---

## 8. Temporal Train / Validation / Test Split

Random splitting is avoided.

The rolling retraining framework uses four chronological snapshots:

```text
Snapshot N      -> Train
Snapshot N+1    -> Train
Snapshot N+2    -> Validation
Snapshot N+3    -> Test
```

This provides a realistic out-of-time estimate of model performance.

---

## 9. Model Training and Tuning

Random Forest and XGBoost were evaluated during development.

Training and tuning use historical training and validation snapshots. The selected model produces a churn probability, while the binary operating threshold is selected from validation data using the Youden J statistic.

The test snapshot remains isolated until the challenger passes validation-based promotion checks.

---

## 10. Azure ML Model Registry

The accepted model is stored in the Azure ML Model Registry together with metadata required for production use and lineage, including:

- Model version
- Hyperparameters
- Operating threshold
- Training snapshot information
- Validation metrics
- Test metrics
- Feature information

The first accepted model becomes the initial champion.

---

## 11. Monthly Batch Scoring

Every month, the latest available transactions are used to construct a new five-month feature window.

```text
Latest Transactions
        |
        v
5-Month Customer Window
        |
        v
14 Features
        |
        v
Champion Model
        |
        v
Churn Probability
        |
        v
Operating Threshold
        |
        v
Predicted Churn Status
```

Monthly scoring is independent of the three-month spacing used for historical training snapshots.

---

## 12. Prediction History

Every monthly prediction is persisted with execution-level lineage.

Typical fields include:

- Customer ID
- Snapshot date
- Churn probability
- Predicted churn
- Model version
- Threshold used
- Azure run ID
- Execution ID
- Scoring timestamp

Storage follows a structure such as:

```text
prediction-history/
└── snapshot-date/
    └── execution-id/
        └── predictions_YYYY-MM-DD.parquet
```

This allows the same historical snapshot to be safely rerun without overwriting previous results.

---

## 13. Immediate Monitoring

Immediate monitoring is available as soon as monthly scoring finishes.

Signals include:

- Feature drift
- Behavioural feature PSI
- Prediction PSI
- Predicted churn rate
- Mean churn probability
- Data-quality checks

Naturally time-dependent features such as `tenure_days` are recorded but excluded from behavioural drift alerts where appropriate.

---

## 14. Delayed Performance Evaluation

A separate delayed-evaluation component compares saved predictions with actual outcomes after the three-month window matures.

```text
June Prediction
      |
      v
Observe Jun-Aug behaviour
      |
      v
September Evaluation
      |
      v
Prediction vs Ground Truth
```

Metrics include:

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall
- F1
- Brier score
- Confusion matrix
- Actual vs predicted churn rate

Results are stored in persistent performance history.

---

## 15. Monitoring and Alerts

Monitoring combines immediate drift signals with delayed performance signals.

Immediate signals include:

- Feature drift
- Behavioural PSI
- Prediction drift

Delayed signals include:

- ROC-AUC degradation
- Recall degradation
- Brier-score deterioration

Alerts are classified as:

```text
OK
WARNING
CRITICAL
```

Alert outputs are persisted for operational review and audit. Monitoring alerts are deliberately non-blocking so that a warning does not prevent monthly scoring outputs from being produced.

---

## 16. Model Retraining

Monthly scoring runs every month, but retraining is scheduled on the cadence at which a new three-month outcome can become fully resolved.

The retraining component first checks whether a new resolved snapshot exists beyond the data already represented by the current champion.

```text
Scheduled retraining period?
        |
        v
New fully resolved snapshot available?
        |
   +----+----+
   |         |
  No        Yes
   |         |
   v         v
 Skip      Train Challenger
```

This prevents unnecessary duplicate retraining.

---

## 17. Champion-Challenger Promotion

When new resolved data is available, a challenger is trained and compared with the current champion using validation data.

Promotion guardrails use:

- ROC-AUC
- Recall
- Brier score

```text
Train Challenger
      |
      v
Compare with Champion
      |
      v
Check Guardrails
      |
  +---+----+
  |        |
Pass      Fail
  |        |
  v        v
Test      Retain
Challenger Champion
  |
  v
Register New Champion Version
```

A challenger is not promoted simply because retraining occurred.

---

## 18. CI/CD and Cloud Security

GitHub manages the source-code lifecycle and controlled deployment process.

### CI

A code push can trigger:

- Python environment setup
- Dependency installation
- Python compilation
- pytest execution

CI does not automatically submit an Azure ML production pipeline.

### CD

Azure ML execution remains manually controlled through a GitHub Actions deployment workflow.

The workflow can accept runtime inputs such as:

- Snapshot date
- Execution mode
- Whether tests should run before deployment

### OIDC Authentication

GitHub generates a short-lived OIDC token for the deployment workflow.

Microsoft Entra ID authenticates the GitHub workload against a federated identity credential.

Azure IAM / RBAC then determines what that authenticated workload is allowed to do inside Azure.

```text
GitHub Actions
      |
      v
OIDC Token
      |
      v
Microsoft Entra ID
Authentication
      |
      v
Azure IAM / RBAC
Authorization
      |
      v
Azure ML Workspace
```

This avoids storing a long-lived Azure client secret in GitHub.

---

## 19. Operational Principles

The architecture follows these principles:

- Temporal integrity before model accuracy
- Out-of-time evaluation rather than random splitting
- Immutable raw-data and execution history
- Monthly scoring with delayed ground-truth evaluation
- Controlled retraining instead of automatic replacement
- Explicit model promotion guardrails
- Independent monitoring of drift and model quality
- Separation of CI from cloud execution
- Secretless GitHub-to-Azure authentication using OIDC
- Reproducible model, prediction, performance, and alert lineage
