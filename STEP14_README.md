# Step 14 - CI/CD and Automated Tests

This package adds a guarded CI/CD layer around the Retail Customer Churn MLOps project.

## What Step 14 implements

### Continuous Integration
`.github/workflows/ci.yml`

Runs automatically on:
- pushes to `main`
- pull requests targeting `main`
- manual workflow runs

CI performs:
1. Python 3.11 setup
2. dependency installation
3. Python compilation/syntax checks
4. pytest automated tests
5. JUnit test-result upload

The tests deliberately target high-risk production contracts:
- PSI WARNING / CRITICAL threshold behavior
- ROC-AUC / recall / Brier regression direction
- preference for held-out test metrics as monitoring reference
- exclusion of `tenure_days` from behavioral-drift alerting
- no delayed-performance comparison against the wrong model version
- delayed churn label remains based on positive-quantity sales only
- Step 13A monitoring artifacts remain present
- persistent prediction/performance/alert history paths remain configured
- all four pipeline components remain wired
- retrain-child output lookup fix remains present

### Controlled CD
`.github/workflows/deploy-azure-ml.yml`

Deployment is deliberately MANUAL using `workflow_dispatch`.

Before Azure submission it:
1. installs dependencies
2. reruns syntax checks
3. reruns pytest
4. authenticates to Azure with OIDC
5. executes `azure/submit_pipeline_job.py`

This prevents every commit from launching an Azure ML production pipeline.

## Installation

Copy these into the repository root:

```text
requirements-ci.txt
pytest.ini
tests/
.github/workflows/
```

Your existing project remains:

```text
azure/
  score_azure.py
  retrain_azure.py
  evaluate_delayed_performance.py
  evaluate_monitoring_alerts.py
  submit_pipeline_job.py
```

## Run locally first

From the repository root:

```powershell
python -m pip install -r requirements-ci.txt
pytest -q
```

Also run:

```powershell
python -m compileall -q azure
```

Do not push CI until local tests pass.

## GitHub CI

Commit the files to GitHub. GitHub Actions automatically discovers workflow files under:

```text
.github/workflows/
```

The CI workflow requires no Azure credentials.

## GitHub CD authentication

The deployment workflow uses OpenID Connect rather than a long-lived Azure client secret.

Configure a Microsoft Entra application or user-assigned managed identity with a federated GitHub credential, then configure GitHub Actions secrets:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
```

The Azure identity should receive only the Azure permissions actually required to submit/read Azure ML jobs and access the workspace resources.

Create a GitHub Environment named:

```text
production
```

Optionally configure required reviewers on that environment. This adds a human approval gate before the Azure deployment job executes.

## Important note about submit_pipeline_job.py

The CD workflow executes your CURRENT working:

```text
azure/submit_pipeline_job.py
```

It does not replace that file.

Therefore keep the working Azure values/configuration in the repository version of the submitter. If you later refactor workspace values to environment variables, the GitHub workflow can supply them without changing the architecture.

## Recommended Git strategy

```text
feature branch
     |
     v
pull request
     |
     v
CI: compile + pytest
     |
     v
merge main
     |
     v
manual "Deploy Azure ML Pipeline"
     |
     v
production approval (optional)
     |
     v
Azure OIDC login
     |
     v
submit_pipeline_job.py
```

This is intentionally simpler than introducing Docker registries, Kubernetes, or complex deployment tooling that this monthly batch ML system does not need.
