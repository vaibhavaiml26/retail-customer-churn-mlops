"""Submit the full Azure ML retail-churn production pipeline.

This version removes the need to hardcode historical scoring job names for
 delayed-performance evaluation. Instead:

1. monthly scoring writes predictions to a persistent datastore location:
   PREDICTION_HISTORY_ROOT/<AS_OF_DATE>/predictions_YYYY-MM-DD.parquet
2. delayed evaluation reads recursively from PREDICTION_HISTORY_ROOT and
   resolves the matured prediction file based on AS_OF_DATE - OUTCOME_MONTHS
3. retraining still runs in parallel and promotion/registration is handled
   explicitly after the pipeline completes.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

import os
import uuid
from datetime import datetime, timezone

# ------------------------------------------------------------------
# Workspace settings
# ------------------------------------------------------------------
SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"

COMPUTE_NAME = "ChurnAmlCompute"
ENVIRONMENT = "azureml:Customer_Churn_Workspace:2"




# Using @latest for DATA assets avoids pinning a broken/stale data version.
# Azure ML resolves @latest to the most recently created asset version.
BOOTSTRAP_DATA = "azureml:Retail_Churn_Data_Bootstrap:1"
REPLAY_DATA = "azureml:Retail_Churn_Data_Monthly:1"

REGISTERED_MODEL_NAME = "retail-churn-xgboost"
# ------------------------------------------------------------------
# Replay / scheduling controls
# ------------------------------------------------------------------

N_TRAIN = 2
N_VAL = 1
N_TEST = 1
STEP_MONTHS = 3

OUTCOME_MONTHS = 3
FIRST_PREDICTION_SNAPSHOT = "2011-06-01"

AS_OF_DATE = os.getenv(
    "AS_OF_DATE",
    "2011-12-01"
)

EXECUTION_ID = os.getenv("EXECUTION_ID")


if not EXECUTION_ID:
    EXECUTION_ID = (
        datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "-"
        + uuid.uuid4().hex[:8]
    )

print("AS_OF_DATE:", AS_OF_DATE)
print("EXECUTION_ID:", EXECUTION_ID)


EXECUTION_MODE = os.getenv(
    "EXECUTION_MODE",
    "full"
)
VALID_MODES = {
    "score_only",
    "score_and_monitor",
    "full",
}

if EXECUTION_MODE not in VALID_MODES:
    raise ValueError(
        f"Invalid EXECUTION_MODE: {EXECUTION_MODE}"
    )
ENABLE_DELAYED_EVAL = (
    EXECUTION_MODE
    in {"score_and_monitor", "full"}
)

ENABLE_MONITORING = (
    EXECUTION_MODE
    in {"score_and_monitor", "full"}
)

ENABLE_RETRAIN = (
    EXECUTION_MODE == "full"
)
# Retraining cadence for the historical replay. The first real retraining
# point is 2011-09-01 and then every 3 months.
RETRAIN_ANCHOR_DATE = "2011-09-01"
RETRAIN_INTERVAL_MONTHS = 3


def _month_index(date_text: str) -> int:
    year, month, _day = [int(part) for part in date_text.split("-")]
    return year * 12 + (month - 1)


def _is_delayed_eval_due(as_of_date: str) -> bool:
    return (
        _month_index(as_of_date)
        >= _month_index(FIRST_PREDICTION_SNAPSHOT) + OUTCOME_MONTHS
    )


def _is_retrain_due(as_of_date: str) -> bool:
    current = _month_index(as_of_date)
    anchor = _month_index(RETRAIN_ANCHOR_DATE)
    return current >= anchor and (current - anchor) % RETRAIN_INTERVAL_MONTHS == 0


RUN_DELAYED_EVAL = _is_delayed_eval_due(AS_OF_DATE)
RUN_RETRAIN = _is_retrain_due(AS_OF_DATE)

print("Delayed evaluation due:", RUN_DELAYED_EVAL)
print("Retraining due:", RUN_RETRAIN)

# ------------------------------------------------------------------
# Persistent datastore locations
# ------------------------------------------------------------------
PREDICTION_HISTORY_ROOT = (
    "azureml://datastores/workspaceblobstore/"
    "paths/retail-churn/prediction-history"
)
PERFORMANCE_HISTORY_ROOT = (
    "azureml://datastores/workspaceblobstore/"
    "paths/retail-churn/performance-history"
)
ALERT_HISTORY_ROOT = (
    "azureml://datastores/workspaceblobstore/"
    "paths/retail-churn/alert-history"
)

# Step 13C alert thresholds. PSI thresholds are operational heuristics; model
# performance thresholds mirror the promotion guardrails already used by the
# project. Critical performance regression is 2x the warning regression.
PREDICTION_PSI_WARNING = 0.10
PREDICTION_PSI_CRITICAL = 0.25
FEATURE_PSI_WARNING = 0.10
FEATURE_PSI_CRITICAL = 0.25
MAX_ROC_AUC_DROP = 0.005
MAX_RECALL_DROP = 0.05
MAX_BRIER_WORSENING = 0.02
CRITICAL_MULTIPLIER = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = PROJECT_ROOT / "azure" / "score_azure.py"
RETRAIN_SCRIPT = PROJECT_ROOT / "azure" / "retrain_azure.py"
DELAYED_EVAL_SCRIPT = PROJECT_ROOT / "azure" / "evaluate_delayed_performance.py"
ALERT_SCRIPT = PROJECT_ROOT / "azure" / "evaluate_monitoring_alerts.py"

for script in [SCORE_SCRIPT, RETRAIN_SCRIPT, DELAYED_EVAL_SCRIPT, ALERT_SCRIPT]:
    if not script.exists():
        raise FileNotFoundError(f"Cannot find required Azure script: {script}")

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

latest_model = ml_client.models.get(name=REGISTERED_MODEL_NAME, label="latest")
CHAMPION_MODEL = f"azureml:{latest_model.name}:{latest_model.version}"
print("Resolved current champion:", CHAMPION_MODEL)

# ------------------------------------------------------------------
# Reusable command components
# ------------------------------------------------------------------
score_component = command(
    name="retail_churn_monthly_score_component",
    display_name="Retail churn monthly scoring",
    description="Reconstruct point-in-time history and batch-score active customers.",
    code=str(PROJECT_ROOT),
    command=(
        "python azure/score_azure.py "
        "--bootstrap-data ${{inputs.bootstrap_data}} "
        "--replay-data ${{inputs.replay_data}} "
        "--model ${{inputs.model}} "
        "--model-asset-ref ${{inputs.model_asset_ref}} "
        "--predictions-output ${{outputs.predictions_output}} "
        "--snapshot-date ${{inputs.snapshot_date}}"
    ),
    inputs={
        "bootstrap_data": Input(type=AssetTypes.URI_FOLDER),
        "replay_data": Input(type=AssetTypes.URI_FOLDER),
        "model": Input(type=AssetTypes.CUSTOM_MODEL),
        "model_asset_ref": Input(type="string"),
        "snapshot_date": Input(type="string"),
    },
    outputs={
        "predictions_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENVIRONMENT,
)

retrain_component = command(
    name="retail_churn_retrain_component",
    display_name="Retail churn retraining check",
    description=(
        "Check whether a newer labeled snapshot has matured; if so train and "
        "evaluate a frozen-parameter XGBoost challenger."
    ),
    code=str(PROJECT_ROOT),
    command=(
        "python azure/retrain_azure.py "
        "--bootstrap-data ${{inputs.bootstrap_data}} "
        "--replay-data ${{inputs.replay_data}} "
        "--champion-model ${{inputs.champion_model}} "
        "--champion-model-ref ${{inputs.champion_model_ref}} "
        "--challenger-model-output ${{outputs.challenger_model}} "
        "--decision-output ${{outputs.decision_output}} "
        "--as-of-date ${{inputs.as_of_date}} "
        "--n-train ${{inputs.n_train}} "
        "--n-val ${{inputs.n_val}} "
        "--n-test ${{inputs.n_test}} "
        "--step-months ${{inputs.step_months}}"
    ),
    inputs={
        "bootstrap_data": Input(type=AssetTypes.URI_FOLDER),
        "replay_data": Input(type=AssetTypes.URI_FOLDER),
        "champion_model": Input(type=AssetTypes.CUSTOM_MODEL),
        "champion_model_ref": Input(type="string"),
        "as_of_date": Input(type="string"),
        "n_train": Input(type="integer"),
        "n_val": Input(type="integer"),
        "n_test": Input(type="integer"),
        "step_months": Input(type="integer"),
    },
    outputs={
        "challenger_model": Output(type=AssetTypes.URI_FOLDER),
        "decision_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENVIRONMENT,
)

delayed_eval_component = command(
    name="retail_churn_delayed_eval_component",
    display_name="Retail churn delayed performance evaluation",
    description=(
        "Evaluate matured historical prediction snapshots using actual "
        "3-month outcomes from persisted prediction history."
    ),
    code=str(PROJECT_ROOT),
    command=(
        "python azure/evaluate_delayed_performance.py "
        "--bootstrap-data ${{inputs.bootstrap_data}} "
        "--replay-data ${{inputs.replay_data}} "
        "--predictions-data ${{inputs.predictions_data}} "
        "--performance-output ${{outputs.performance_output}} "
        "--as-of-date ${{inputs.as_of_date}} "
        "--outcome-months ${{inputs.outcome_months}} "
        "--first-prediction-snapshot ${{inputs.first_prediction_snapshot}}"
    ),
    inputs={
        "bootstrap_data": Input(type=AssetTypes.URI_FOLDER),
        "replay_data": Input(type=AssetTypes.URI_FOLDER),
        "predictions_data": Input(type=AssetTypes.URI_FOLDER),
        "as_of_date": Input(type="string"),
        "outcome_months": Input(type="integer"),
        "first_prediction_snapshot": Input(type="string"),
    },
    outputs={
        "performance_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENVIRONMENT,
)


# Step 13C uses the same Python alert evaluator in two component shapes so we
# avoid optional URI-folder input semantics in the Azure command definition.
alert_component_immediate = command(
    name="retail_churn_monitoring_alert_component",
    display_name="Retail churn monitoring alerts",
    description="Evaluate immediate drift/monitoring signals for the monthly score.",
    code=str(PROJECT_ROOT),
    command=(
        "python azure/evaluate_monitoring_alerts.py "
        "--score-data ${{inputs.score_data}} "
        "--model ${{inputs.model}} "
        "--model-asset-ref ${{inputs.model_asset_ref}} "
        "--alert-output ${{outputs.alert_output}} "
        "--prediction-psi-warning ${{inputs.prediction_psi_warning}} "
        "--prediction-psi-critical ${{inputs.prediction_psi_critical}} "
        "--feature-psi-warning ${{inputs.feature_psi_warning}} "
        "--feature-psi-critical ${{inputs.feature_psi_critical}} "
        "--max-roc-auc-drop ${{inputs.max_roc_auc_drop}} "
        "--max-recall-drop ${{inputs.max_recall_drop}} "
        "--max-brier-worsening ${{inputs.max_brier_worsening}} "
        "--critical-multiplier ${{inputs.critical_multiplier}}"
    ),
    inputs={
        "score_data": Input(type=AssetTypes.URI_FOLDER),
        "model": Input(type=AssetTypes.CUSTOM_MODEL),
        "model_asset_ref": Input(type="string"),
        "prediction_psi_warning": Input(type="number"),
        "prediction_psi_critical": Input(type="number"),
        "feature_psi_warning": Input(type="number"),
        "feature_psi_critical": Input(type="number"),
        "max_roc_auc_drop": Input(type="number"),
        "max_recall_drop": Input(type="number"),
        "max_brier_worsening": Input(type="number"),
        "critical_multiplier": Input(type="number"),
    },
    outputs={
        "alert_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENVIRONMENT,
)

alert_component_with_performance = command(
    name="retail_churn_monitoring_alert_with_performance_component",
    display_name="Retail churn monitoring alerts",
    description=(
        "Evaluate immediate drift signals plus matured delayed model performance."
    ),
    code=str(PROJECT_ROOT),
    command=(
        "python azure/evaluate_monitoring_alerts.py "
        "--score-data ${{inputs.score_data}} "
        "--performance-data ${{inputs.performance_data}} "
        "--model ${{inputs.model}} "
        "--model-asset-ref ${{inputs.model_asset_ref}} "
        "--alert-output ${{outputs.alert_output}} "
        "--prediction-psi-warning ${{inputs.prediction_psi_warning}} "
        "--prediction-psi-critical ${{inputs.prediction_psi_critical}} "
        "--feature-psi-warning ${{inputs.feature_psi_warning}} "
        "--feature-psi-critical ${{inputs.feature_psi_critical}} "
        "--max-roc-auc-drop ${{inputs.max_roc_auc_drop}} "
        "--max-recall-drop ${{inputs.max_recall_drop}} "
        "--max-brier-worsening ${{inputs.max_brier_worsening}} "
        "--critical-multiplier ${{inputs.critical_multiplier}}"
    ),
    inputs={
        "score_data": Input(type=AssetTypes.URI_FOLDER),
        "performance_data": Input(type=AssetTypes.URI_FOLDER),
        "model": Input(type=AssetTypes.CUSTOM_MODEL),
        "model_asset_ref": Input(type="string"),
        "prediction_psi_warning": Input(type="number"),
        "prediction_psi_critical": Input(type="number"),
        "feature_psi_warning": Input(type="number"),
        "feature_psi_critical": Input(type="number"),
        "max_roc_auc_drop": Input(type="number"),
        "max_recall_drop": Input(type="number"),
        "max_brier_worsening": Input(type="number"),
        "critical_multiplier": Input(type="number"),
    },
    outputs={
        "alert_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENVIRONMENT,
)


@dsl.pipeline(
    compute=COMPUTE_NAME,
    description=(
        "Retail churn production cycle: monthly batch scoring, delayed "
        "performance evaluation when mature, and quarterly retraining when due."
    ),
)
def retail_churn_monthly_pipeline(
    bootstrap_data,
    replay_data,
    champion_model,
    champion_model_ref,
    as_of_date,
    n_train,
    n_val,
    n_test,
    step_months,
    outcome_months,
    first_prediction_snapshot,
    prediction_psi_warning,
    prediction_psi_critical,
    feature_psi_warning,
    feature_psi_critical,
    max_roc_auc_drop,
    max_recall_drop,
    max_brier_worsening,
    critical_multiplier,
):
    # Monthly scoring always runs.
    score_job = score_component(
        bootstrap_data=bootstrap_data,
        replay_data=replay_data,
        model=champion_model,
        model_asset_ref=champion_model_ref,
        snapshot_date=as_of_date,
    )
    score_job.name = "monthly_score"

    pipeline_outputs = {
        "predictions_output": score_job.outputs.predictions_output,
    }

    # Delayed performance starts only when the first prediction has completed
    # its full forward outcome window. With first prediction=2011-06-01 and a
    # 3-month outcome, the first delayed evaluation is 2011-09-01.
    if RUN_DELAYED_EVAL:
        delayed_eval_job = delayed_eval_component(
            bootstrap_data=bootstrap_data,
            replay_data=replay_data,
            predictions_data=Input(
                type=AssetTypes.URI_FOLDER,
                path=PREDICTION_HISTORY_ROOT,
                mode="download",
            ),
            as_of_date=as_of_date,
            outcome_months=outcome_months,
            first_prediction_snapshot=first_prediction_snapshot,
        )
        delayed_eval_job.name = "delayed_evaluation"
        pipeline_outputs["performance_output"] = (
            delayed_eval_job.outputs.performance_output
        )

        # When delayed ground truth is mature, monitoring includes production
        # performance as well as immediate drift signals.
        alert_job = alert_component_with_performance(
            score_data=score_job.outputs.predictions_output,
            performance_data=delayed_eval_job.outputs.performance_output,
            model=champion_model,
            model_asset_ref=champion_model_ref,
            prediction_psi_warning=prediction_psi_warning,
            prediction_psi_critical=prediction_psi_critical,
            feature_psi_warning=feature_psi_warning,
            feature_psi_critical=feature_psi_critical,
            max_roc_auc_drop=max_roc_auc_drop,
            max_recall_drop=max_recall_drop,
            max_brier_worsening=max_brier_worsening,
            critical_multiplier=critical_multiplier,
        )
    else:
        # Before delayed labels mature, monitor only immediate scoring/drift
        # health. This still makes alerting available from the very first month.
        alert_job = alert_component_immediate(
            score_data=score_job.outputs.predictions_output,
            model=champion_model,
            model_asset_ref=champion_model_ref,
            prediction_psi_warning=prediction_psi_warning,
            prediction_psi_critical=prediction_psi_critical,
            feature_psi_warning=feature_psi_warning,
            feature_psi_critical=feature_psi_critical,
            max_roc_auc_drop=max_roc_auc_drop,
            max_recall_drop=max_recall_drop,
            max_brier_worsening=max_brier_worsening,
            critical_multiplier=critical_multiplier,
        )

    alert_job.name = "monitoring_alerts"
    pipeline_outputs["alert_output"] = alert_job.outputs.alert_output

    # Retraining is inserted into the Azure DAG only on the quarterly cadence.
    # retrain_azure.py still performs its internal readiness checks as a second
    # safety layer in case the required labels/data are not actually available.
    if RUN_RETRAIN:
        retrain_job = retrain_component(
            bootstrap_data=bootstrap_data,
            replay_data=replay_data,
            champion_model=champion_model,
            champion_model_ref=champion_model_ref,
            as_of_date=as_of_date,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            step_months=step_months,
        )
        retrain_job.name = "retrain_check"
        pipeline_outputs["challenger_model"] = retrain_job.outputs.challenger_model
        pipeline_outputs["decision_output"] = retrain_job.outputs.decision_output

    return pipeline_outputs


pipeline_job = retail_churn_monthly_pipeline(
    bootstrap_data=Input(
        type=AssetTypes.URI_FOLDER,
        path=BOOTSTRAP_DATA,
        mode="download",
    ),
    replay_data=Input(
        type=AssetTypes.URI_FOLDER,
        path=REPLAY_DATA,
        mode="download",
    ),
    champion_model=Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=CHAMPION_MODEL,
        mode="download",
    ),
    champion_model_ref=CHAMPION_MODEL,
    as_of_date=AS_OF_DATE,
    n_train=N_TRAIN,
    n_val=N_VAL,
    n_test=N_TEST,
    step_months=STEP_MONTHS,
    outcome_months=OUTCOME_MONTHS,
    first_prediction_snapshot=FIRST_PREDICTION_SNAPSHOT,
    prediction_psi_warning=PREDICTION_PSI_WARNING,
    prediction_psi_critical=PREDICTION_PSI_CRITICAL,
    feature_psi_warning=FEATURE_PSI_WARNING,
    feature_psi_critical=FEATURE_PSI_CRITICAL,
    max_roc_auc_drop=MAX_ROC_AUC_DROP,
    max_recall_drop=MAX_RECALL_DROP,
    max_brier_worsening=MAX_BRIER_WORSENING,
    critical_multiplier=CRITICAL_MULTIPLIER,
)

pipeline_job.display_name = f"retail-churn-production-{AS_OF_DATE}"
pipeline_job.experiment_name = "retail-churn-production-pipeline"

# Persist score outputs into prediction history so later pipeline runs can
# evaluate delayed performance without hardcoding prior Azure job names.
prediction_history_path=(
    f"{PREDICTION_HISTORY_ROOT.rstrip('/')}/"
    f"{AS_OF_DATE}/"
    f"{EXECUTION_ID}"
)
performance_history_path =(
    f"{PERFORMANCE_HISTORY_ROOT.rstrip('/')}/"
    f"{AS_OF_DATE}/"
    f"{EXECUTION_ID}"
)
alert_history_path=(
    f"{ALERT_HISTORY_ROOT.rstrip('/')}/"
    f"{AS_OF_DATE}/"
    f"{EXECUTION_ID}"
                  )
#prediction_history_path = f"{PREDICTION_HISTORY_ROOT.rstrip('/')}/{AS_OF_DATE}"
#performance_history_path = f"{PERFORMANCE_HISTORY_ROOT.rstrip('/')}/{AS_OF_DATE}"
#alert_history_path = f"{ALERT_HISTORY_ROOT.rstrip('/')}/{AS_OF_DATE}"

# The score/delayed-evaluation/alert child outputs are promoted to pipeline-level
# outputs by the pipeline return dictionary above. Pin the *pipeline-level*
# outputs to persistent datastore paths. This is the supported place to
# customize the final output location of promoted outputs.
pipeline_job.outputs.predictions_output.path = prediction_history_path
pipeline_job.outputs.predictions_output.mode = "upload"

if RUN_DELAYED_EVAL:
    pipeline_job.outputs.performance_output.path = performance_history_path
    pipeline_job.outputs.performance_output.mode = "upload"

pipeline_job.outputs.alert_output.path = alert_history_path
pipeline_job.outputs.alert_output.mode = "upload"

returned_job = ml_client.jobs.create_or_update(pipeline_job)

print("\nPipeline submitted:", returned_job.name)
print("Studio URL:", returned_job.studio_url)
print("Champion model:", CHAMPION_MODEL)
print("Prediction history target:", prediction_history_path)
print("Performance history target:", performance_history_path)
print("Alert history target:", alert_history_path)
print(
    "Predictions output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/predictions_output",
)
if RUN_DELAYED_EVAL:
    print(
        "Performance output URI:",
        f"azureml://jobs/{returned_job.name}/outputs/performance_output",
    )
else:
    print("Delayed evaluation not due for this as-of date.")

print(
    "Alert output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/alert_output",
)

if RUN_RETRAIN:
    print(
        "Decision output URI:",
        f"azureml://jobs/{returned_job.name}/outputs/decision_output",
    )
    print(
        "Challenger output URI:",
        f"azureml://jobs/{returned_job.name}/outputs/challenger_model",
    )
else:
    print("Retraining not due for this as-of date; no retrain child job submitted.")

terminal_states = {"Completed", "Failed", "Canceled", "NotResponding"}
while True:
    current_job = ml_client.jobs.get(returned_job.name)
    print("Pipeline status:", current_job.status)
    if current_job.status in terminal_states:
        break
    time.sleep(20)

if current_job.status != "Completed":
    raise RuntimeError(
        f"Pipeline ended with status {current_job.status}. "
        f"Check Azure ML Studio: {returned_job.studio_url}"
    )

print("\nAzure ML pipeline completed successfully.")

# ------------------------------------------------------------------
# Read the monitoring alert summary from the monitoring child job.
# Alert download failure is non-fatal because the Azure pipeline outputs are
# already persisted under ALERT_HISTORY_ROOT.
# ------------------------------------------------------------------
try:
    child_jobs_for_alert = list(
        ml_client.jobs.list(parent_job_name=returned_job.name)
    )
    alert_candidates = [
        child
        for child in child_jobs_for_alert
        if (child.display_name or "").strip().lower() == "monitoring_alerts"
    ]
    if len(alert_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one monitoring_alerts child; "
            f"found {len(alert_candidates)}."
        )

    alert_child = ml_client.jobs.get(alert_candidates[0].name)
    with tempfile.TemporaryDirectory(prefix="churn_pipeline_alert_") as tmp:
        ml_client.jobs.download(
            name=alert_child.name,
            download_path=tmp,
            output_name="alert_output",
        )
        alert_files = list(Path(tmp).rglob("alert_summary.json"))
        alert_summary = json.loads(
            alert_files[0].read_text(encoding="utf-8")
        )

    print("\nMonitoring alert summary")
    print("=" * 70)
    print(json.dumps(alert_summary, indent=2))
except Exception as exc:
    print("\nMonitoring alerts were persisted in Azure, but the local SDK")
    print("could not download alert_output for display.")
    print("Alert download error:", repr(exc))
    print("Inspect:", alert_history_path)

# If this is not a quarterly retraining month, there is deliberately no
# retraining child job, no promotion decision, and nothing to register.
if not RUN_RETRAIN:
    print("Retraining was not scheduled for this as-of date.")
    print("Current champion remains:", CHAMPION_MODEL)
    raise SystemExit(0)

# ------------------------------------------------------------------
# Explicit post-pipeline promotion/registration action
# ------------------------------------------------------------------

# List child jobs, identify the retraining node by the display name that we
# assigned in the pipeline DAG, then retrieve the full child job resource.
child_jobs = list(
    ml_client.jobs.list(parent_job_name=returned_job.name)
)

print("\nPipeline child jobs")
print("=" * 70)

retrain_candidates = []
for child in child_jobs:
    print(
        f"name={child.name}, "
        f"display_name={child.display_name}, "
        f"status={child.status}"
    )

    if (child.display_name or "").strip().lower() == "retrain_check":
        retrain_candidates.append(child)

if len(retrain_candidates) != 1:
    raise RuntimeError(
        "Retraining was expected, but exactly one child job with "
        "display_name='retrain_check' was required; "
        f"found {len(retrain_candidates)}."
    )

# jobs.list() is sufficient for discovery, but retrieve the full child job
# resource before operating on its named outputs.
retrain_child = ml_client.jobs.get(retrain_candidates[0].name)

print("\nResolved retraining child job")
print("-" * 70)
print("Name:", retrain_child.name)
print("Display name:", retrain_child.display_name)
print("Status:", retrain_child.status)

try:
    with tempfile.TemporaryDirectory(prefix="churn_pipeline_decision_") as tmp:
        ml_client.jobs.download(
            name=retrain_child.name,
            download_path=tmp,
            output_name="decision_output",
        )

        decision_files = list(Path(tmp).rglob("promotion_decision.json"))
        if len(decision_files) != 1:
            raise RuntimeError(
                "Expected exactly one promotion_decision.json from the "
                f"retraining child; found {len(decision_files)}"
            )

        decision = json.loads(
            decision_files[0].read_text(encoding="utf-8")
        )
except Exception as exc:
    print(
        "\nPipeline completed, but decision_output could not be downloaded "
        "from the retraining child."
    )
    print("No model will be registered automatically. Safe failure behavior.")
    print("Download error:", repr(exc))
    print("Retraining child job:", retrain_child.name)
    raise SystemExit(0)

print("\nPromotion decision")
print("=" * 70)
print(json.dumps(decision, indent=2))

if decision.get("retrain_skipped"):
    print("\nRetraining skipped:", decision.get("reason"))
    print("Current champion remains:", CHAMPION_MODEL)
    raise SystemExit(0)

if not decision.get("promote"):
    print("\nChallenger rejected:", decision.get("reason"))
    print("Current champion remains:", CHAMPION_MODEL)
    raise SystemExit(0)

# Register the model from the RETRAINING CHILD job output, not from the parent
# pipeline output. This is the child that actually produced challenger_model.
model_path = (
    f"azureml://jobs/{retrain_child.name}/outputs/challenger_model"
)

print("\nRegistering promoted challenger from:")
print(model_path)

run_model = Model(
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    name=REGISTERED_MODEL_NAME,
    description=(
        "Retail churn XGBoost challenger promoted from pipeline "
        f"{returned_job.name}, retraining child {retrain_child.name}, "
        f"as of {AS_OF_DATE}."
    ),
    tags={
        "promotion": "promoted",
        "as_of_date": AS_OF_DATE,
        "previous_champion": CHAMPION_MODEL,
        "source_pipeline_job": returned_job.name,
        "source_retrain_job": retrain_child.name,
    },
)
registered = ml_client.models.create_or_update(run_model)
new_model_ref = f"azureml:{registered.name}:{registered.version}"

print("\nPromoted challenger registered successfully")
print("Previous champion:", CHAMPION_MODEL)
print("New champion:", new_model_ref)
print("The NEXT monthly pipeline run will resolve and use this new latest version.")
