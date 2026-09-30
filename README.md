# Retail Customer Churn Prediction & MLOps

## Project Overview

This project builds an end-to-end **customer churn prediction and MLOps system** for a grocery retailer using almost two years of historical transaction data.

The source data was transactional, containing invoices, purchased items, quantities, prices, returns, customer identifiers, and transaction dates. The business objective, however, was customer-centric:

> Given a customer's historical purchasing behaviour, estimate the probability that the customer will become inactive over the following three months.

The project therefore required more than training a classification algorithm. Key challenges included defining churn for a non-subscription retail business, converting transaction-level data into customer-level behavioural features, designing leakage-free temporal snapshots, selecting an operating threshold, enabling monthly batch scoring, evaluating predictions only after outcomes matured, monitoring drift and performance, and implementing controlled champion-challenger retraining.

The final system supports:

- Customer-level feature engineering from raw retail transactions
- Temporal training snapshots with forward-looking churn labels
- Out-of-time validation and testing
- Random Forest and XGBoost model evaluation
- Validation-based threshold selection using Youden J
- Azure Machine Learning training, compute, storage, pipelines, and model registry
- Monthly batch scoring
- Persistent prediction history
- Delayed ground-truth performance evaluation
- Feature and prediction drift monitoring
- Champion-challenger retraining and promotion guardrails
- GitHub CI/CD with automated tests and OIDC-based Azure authentication

---

## 1. Defining Customer Churn

The first challenge was defining **churn** in a grocery-retail environment.

Unlike a subscription business, there is no explicit cancellation event. A customer usually does not declare that they have churned. Churn therefore had to be inferred from purchasing behaviour.

Historical transaction patterns were analysed to identify an inactivity period that could be used as a meaningful churn definition. The final target was based on a **three-month forward outcome window**.

For a prediction snapshot at time `T`:

```text
Historical customer behaviour
          |
          v
          T
          |
          +-------- Next 3 months --------+
          |                               |
          |      Any positive sale?       |
          |                               |
          +-- Yes -> Active = 0           |
          |                               |
          +-- No  -> Churned = 1          |
```

Only positive-quantity sale activity is considered when determining whether a customer remains active. Returns or cancellation transactions by themselves do not make a customer active.

Conceptually:

```text
Customer makes >= 1 valid sale in next 3 months
        -> inactive_90d = 0

Customer makes no valid sale in next 3 months
        -> inactive_90d = 1
```

This converts an otherwise ambiguous business concept into a reproducible supervised-learning target.

---

## 2. Converting Transaction Data into Customer Features

The second major challenge was the mismatch between the available data and the required prediction unit.

The retailer supplied **transaction-level data**, while the model needed to predict churn at the **customer level**.

Raw data looked conceptually like:

```text
Customer | Invoice | Date | Item | Quantity | Price
---------------------------------------------------
C001     | I101    | Jan  | A    | 2        | ...
C001     | I101    | Jan  | B    | 1        | ...
C001     | I205    | Feb  | C    | 3        | ...
C002     | I310    | Feb  | A    | 1        | ...
```

The model instead required one customer feature record per snapshot:

```text
Customer | Revenue | Transactions | Recency | Returns | Tenure | ...
-------------------------------------------------------------------
C001     | ...     | ...          | ...     | ...     | ...    |
C002     | ...     | ...          | ...     | ...     | ...    |
```

The final production feature set contains **14 customer behavioural features**:

1. `total_sales_revenue`
2. `total_items_purchased`
3. `total_sale_txs`
4. `avg_sale_days`
5. `recency`
6. `total_return_amount`
7. `total_items_returned`
8. `total_return_txs`
9. `avg_return_days`
10. `recent_revenue`
11. `tenure_days`
12. `recent_tx_count`
13. `revenue_momentum`
14. `tx_count_momentum`

These features capture purchasing value, transaction frequency, recency, returns, customer tenure, recent activity, and behavioural momentum.

---

## 3. Temporal Snapshot Design

Temporal snapshot construction was one of the most important modelling decisions.

For each historical training example, the system separates:

```text
Feature Window
      +
Future Outcome Window
```

After experimentation and analysis, a **five-month feature window** was selected.

```text
<--------- 5 months ---------><------ 3 months ------>
       FEATURE WINDOW                 OUTCOME WINDOW

Customer behaviour                    Was customer
used to calculate                     active or inactive?
14 features
```

The model receives only information available before the snapshot date and predicts what happens afterwards. This separation is critical to preventing future-data leakage.

---

## 4. Balancing Snapshot Overlap and Training Data

Almost two years of transaction data becomes relatively limited once every observation requires both a five-month feature window and a three-month future outcome window.

Creating snapshots too frequently would make adjacent training samples highly correlated. For example, monthly five-month windows would share four months of transaction history.

On the other hand, spacing snapshots too far apart would reduce the already limited training data.

The project therefore balanced:

```text
Reduce temporal overlap
          ^
          |
          v
Maximise available training data
```

Snapshots were spaced **three months apart**, which results in a **two-month overlap** between adjacent five-month feature windows.

```text
Snapshot 1
[ M1 M2 M3 M4 M5 ]

Snapshot 2
         [ M4 M5 M6 M7 M8 ]

Overlap
         [ M4 M5 ]
```

This provided a practical compromise between sample independence and available training volume.

---

## 5. Training Observation Structure

Each historical training snapshot consists of:

```text
5-month customer feature window
            +
3-month future outcome window
```

A single modelling row is therefore conceptually:

```text
CustomerID
SnapshotDate
Feature1
Feature2
...
Feature14
inactive_90d
```

The natural observation key is:

```text
(CustomerID, SnapshotDate)
```

The same customer can appear in multiple snapshots because each snapshot represents the customer's state at a different point in time.

---

## 6. Temporal Training, Validation and Test Design

Random train/test splitting was deliberately avoided because it can mix later customer behaviour into training while earlier periods appear in validation or testing.

Instead, the data was split chronologically.

For the rolling retraining framework, four consecutive resolved snapshots are used as:

```text
Snapshot N      -> Training
Snapshot N+1    -> Training
Snapshot N+2    -> Validation
Snapshot N+3    -> Test
```

This produces a realistic out-of-time evaluation and better represents how the model behaves in production.

The held-out test snapshot is evaluated only after the challenger has passed validation-based promotion checks.

---

## 7. Monthly Prediction Cadence

Historical training snapshots are spaced three months apart to reduce overlap, but **production scoring occurs monthly**.

At the completion of each month:

```text
Latest available transactions
              |
              v
Build latest 5-month feature window
              |
              v
Generate 14 customer features
              |
              v
Load current champion model
              |
              v
Generate churn probability
              |
              v
Apply operating threshold
              |
              v
Persist predictions
```

This allows the business to receive refreshed churn predictions every month even though training examples are spaced farther apart.

---

## 8. Prediction History and Delayed Ground Truth

Model performance cannot be measured immediately because the churn target requires three months of future behaviour.

Predictions are therefore persisted for later evaluation.

```text
June prediction
      |
      v
Prediction history
      |
      | wait for outcome window
      v
September
      |
      v
Actual Jun-Aug transaction activity
      |
      v
Generate true churn labels
      |
      v
Compare prediction vs actual
```

This leads to two forms of monitoring.

### Immediate monitoring

Available as soon as monthly scoring completes:

- Feature drift
- Prediction drift
- Predicted churn rate
- Probability distribution
- Data-quality checks

### Delayed performance monitoring

Available once the three-month outcome period has matured:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Accuracy
- Brier score
- Confusion matrix
- Actual vs predicted churn rate

---

## 9. Probability Threshold Selection

The model produces a churn probability rather than a direct binary decision.

```text
Customer A -> P(churn) = 0.18
Customer B -> P(churn) = 0.42
Customer C -> P(churn) = 0.79
```

An operating threshold converts the probability into a classification:

```text
Probability >= threshold -> Predict churn
Probability < threshold  -> Predict active
```

The threshold was selected using the **Youden J statistic** on validation data. Test data was not used for threshold tuning.

The selected threshold is stored with the registered model metadata and reused during production scoring.

---

## 10. Model Evaluation and Promotion Metrics

Multiple metrics are used because no single metric fully describes churn-model quality.

### ROC-AUC
Measures ranking ability across all decision thresholds.

### Recall
Measures the proportion of actual churners correctly identified.

### F1 Score
Balances precision and recall and is reported as an important classification metric.

### PR-AUC
Provides additional insight when class distributions are imbalanced.

### Brier Score
Measures the quality of the predicted probabilities themselves.

For automated champion-challenger retraining, the promotion guardrails use:

- ROC-AUC
- Recall
- Brier score

A challenger is allowed only limited degradation against the current champion. If the challenger breaches the promotion guardrails, the existing champion remains deployed.

---

## 11. Champion-Challenger Retraining

The system follows a **champion-challenger** retraining strategy.

```text
New resolved historical data
          |
          v
Train challenger
          |
          v
Evaluate on forward validation snapshot
          |
          v
Compare against champion
          |
     +----+----+
     |         |
   Pass       Fail
     |         |
     v         v
Evaluate     Reject
on test      challenger
     |
     v
Register new champion version
```

Retraining is scheduled only when an outcome window can mature, and the retraining component independently checks whether a genuinely new fully resolved snapshot exists before training another challenger.

---

## 12. CI/CD and MLOps Implementation

Azure Machine Learning and GitHub are used together to provide the production MLOps foundation.

### Azure Machine Learning

Azure ML is used for:

- Managed compute
- Data and output storage
- Model training
- Pipeline orchestration
- Model registry and versioning
- Monthly batch scoring
- Prediction and performance history
- Monitoring outputs
- Champion-challenger retraining

### GitHub

GitHub is used for:

- Source-code management
- Version control
- Continuous Integration
- Automated tests
- Controlled Continuous Deployment
- Azure authentication orchestration using OIDC

The CI/CD flow is:

```text
Local Development
      |
      v
Git Commit / Push
      |
      v
GitHub Repository
      |
      v
Continuous Integration
      |
      +-- Set up Python
      +-- Install dependencies
      +-- Compile Python code
      +-- Run pytest
      |
      v
Code validated
      |
      | manual deployment trigger
      v
GitHub CD Workflow
      |
      v
GitHub OIDC Token
      |
      v
Microsoft Entra ID
      |
      v
Azure IAM / RBAC
      |
      v
Azure Machine Learning
      |
      v
Submit ML Pipeline
```

### Authentication and Authorization

GitHub generates a short-lived OIDC token for the deployment workflow. Microsoft Entra ID validates the GitHub workload against a federated identity credential.

This removes the need to store a long-lived Azure client secret in GitHub.

The responsibilities are separated as follows:

```text
Microsoft Entra ID
"Who are you?"
        |
        +-- Authentication

Azure IAM / RBAC
"What are you allowed to do?"
        |
        +-- Authorization
```

The GitHub deployment identity is granted only the Azure permissions required to submit and interact with Azure ML workloads.

### Separation of CI and CD

A normal code push can trigger CI tests, but it does **not** automatically run the Azure ML production pipeline.

Actual Azure execution remains manually controlled using a GitHub Actions `workflow_dispatch` deployment workflow.

This allows source-code changes to be validated without unnecessarily starting Azure compute, scoring, monitoring, or retraining jobs.

---

## 13. Reproducible Pipeline Executions

Every Azure ML execution receives a unique execution identifier.

Outputs are stored using both:

```text
Snapshot Date
+
Execution ID
```

For example:

```text
prediction-history/
└── 2011-09-01/
    ├── execution-1/
    │   └── predictions_2011-09-01.parquet
    └── execution-2/
        └── predictions_2011-09-01.parquet
```

The same partitioning concept is used for prediction, performance, and alert history.

This supports auditability, safe reruns, debugging, lineage, and reproducibility without overwriting earlier results.

---

## 14. Technology Stack

- **Python 3.11**
- **pandas / NumPy**
- **scikit-learn**
- **XGBoost**
- **Azure Machine Learning SDK v2**
- **Azure ML Compute / Data Assets / Model Registry / Pipelines**
- **Git / GitHub**
- **GitHub Actions**
- **pytest**
- **Microsoft Entra ID**
- **OIDC workload identity federation**
- **Azure IAM / RBAC**

---

## 15. Key Design Principles

The project was designed around the following principles:

- Prevent future-data leakage in all temporal datasets
- Evaluate models using out-of-time validation and test snapshots
- Separate historical training cadence from monthly production scoring
- Persist predictions so performance can be evaluated after outcomes mature
- Monitor both data/prediction drift and delayed model performance
- Separate retraining from hyperparameter retuning
- Use champion-challenger governance instead of automatically replacing the production model
- Keep CI separate from cloud deployment
- Use secretless OIDC authentication between GitHub and Azure
- Preserve immutable execution history for reproducibility and auditability

---

## Repository Notes

This repository intentionally excludes secrets, Azure subscription identifiers, tenant identifiers, local machine paths, raw customer data, and generated model/output artifacts.

Public examples and documentation describe the production design without exposing private infrastructure or source data.
