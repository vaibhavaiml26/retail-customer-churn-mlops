"""Step 11: submit one Azure ML pipeline that runs monthly scoring and
retraining-readiness/champion-challenger evaluation together.

The two child jobs intentionally run independently against the SAME champion
that existed when the pipeline was submitted:

* monthly_score always scores the requested as-of date;
* quarterly_retrain runs every month too, but retrain_azure.py now exits
  cheaply unless a newer fully-resolved labeled snapshot exists.

If retraining produces promote=true, this submitter registers the challenger
only AFTER the Azure pipeline completes. The newly registered model therefore
becomes the champion for the NEXT monthly pipeline run. This keeps registration
as an explicit orchestration action instead of hiding production-state changes
inside the training container.
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

# ---------------------------------------------------------------------
# Fill these with the same values already used by your Step 9/10 scripts.
# ---------------------------------------------------------------------
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

# First Step-11 validation run. It should score October with champion v2 and
# SKIP retraining because snapshot 6 is not fully resolved until November data
# has arrived (i.e. the 2011-12-01 as-of run).
AS_OF_DATE = "2011-12-01"

# Historical replay split used for production-simulation retraining.
N_TRAIN = 2
N_VAL = 1
N_TEST = 1
STEP_MONTHS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = PROJECT_ROOT / "azure" / "score_azure.py"
RETRAIN_SCRIPT = PROJECT_ROOT / "azure" / "retrain_azure.py"

for script in [SCORE_SCRIPT, RETRAIN_SCRIPT]:
    if not script.exists():
        raise FileNotFoundError(f"Cannot find required Azure script: {script}")

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# Resolve the exact current champion version ONCE at pipeline submission time.
# This preserves exact lineage in scoring/retraining metadata. Only promoted
# challengers are registered under this model name, so latest == champion in
# this project design.
latest_model = ml_client.models.get(name=REGISTERED_MODEL_NAME, label="latest")
CHAMPION_MODEL = f"azureml:{latest_model.name}:{latest_model.version}"
print("Resolved current champion:", CHAMPION_MODEL)


# ---------------------------------------------------------------------
# Reusable command components
# ---------------------------------------------------------------------
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


@dsl.pipeline(
    compute=COMPUTE_NAME,
    description=(
        "Retail churn production cycle: monthly batch scoring plus a safe "
        "retraining-readiness/champion-challenger check."
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
):
    # Scoring and retraining use the champion that existed at the start of this
    # pipeline. They can run in parallel because neither consumes the other's
    # output. A promoted challenger is registered after the pipeline completes
    # and is picked up by the next monthly run.
    score_job = score_component(
        bootstrap_data=bootstrap_data,
        replay_data=replay_data,
        model=champion_model,
        model_asset_ref=champion_model_ref,
        snapshot_date=as_of_date,
    )
    score_job.name = "monthly_score"

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

    return {
        "predictions_output": score_job.outputs.predictions_output,
        "challenger_model": retrain_job.outputs.challenger_model,
        "decision_output": retrain_job.outputs.decision_output,
    }


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
)

pipeline_job.display_name = f"retail-churn-production-{AS_OF_DATE}"
pipeline_job.experiment_name = "retail-churn-production-pipeline"

returned_job = ml_client.jobs.create_or_update(pipeline_job)

print("\nPipeline submitted:", returned_job.name)
print("Studio URL:", returned_job.studio_url)
print(
    "Predictions output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/predictions_output",
)
print(
    "Decision output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/decision_output",
)
print(
    "Challenger output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/challenger_model",
)

# Continue using status polling rather than jobs.stream(), because this local
# machine previously hit a workspace-storage SAS error while streaming logs.
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

# # ---------------------------------------------------------------------
# # Explicit post-pipeline promotion/registration action
# # ---------------------------------------------------------------------
#
# # Pipeline component outputs are most reliably downloaded from the
# # child job that produced them.
# child_jobs = list(
#     ml_client.jobs.list(parent_job_name=returned_job.name)
# )
#
# print("\nPipeline child jobs")
# print("=" * 70)
#
# for child in child_jobs:
#     print(
#         f"name={child.name}, "
#         f"display_name={child.display_name}, "
#         f"status={child.status}"
#     )
#
# # Locate the retraining component.
# retrain_candidates = [
#     child
#     for child in child_jobs
#     if "retrain" in (
#         f"{child.name or ''} {child.display_name or ''}"
#     ).lower()
# ]
#
# if len(retrain_candidates) != 1:
#     raise RuntimeError(
#         "Expected exactly one retraining child job; "
#         f"found {len(retrain_candidates)}. "
#         f"Children: "
#         f"{[(j.name, j.display_name) for j in child_jobs]}"
#     )
#
# retrain_child = retrain_candidates[0]
#
# print("\nResolved retraining child job:")
# print("  Name:", retrain_child.name)
# print("  Display name:", retrain_child.display_name)
# print("  Status:", retrain_child.status)
#
#
# # -------------------------------------------------------------
# # Download decision directly from retraining CHILD job
# # -------------------------------------------------------------
# try:
#     with tempfile.TemporaryDirectory(
#         prefix="churn_pipeline_"
#     ) as tmp:
#
#         ml_client.jobs.download(
#             name=retrain_child.name,
#             download_path=tmp,
#             output_name="decision_output",
#         )
#
#         decision_files = list(
#             Path(tmp).rglob("promotion_decision.json")
#         )
#
#         if len(decision_files) != 1:
#             raise RuntimeError(
#                 "Expected exactly one promotion_decision.json "
#                 "from retraining child job; "
#                 f"found {len(decision_files)}"
#             )
#
#         decision = json.loads(
#             decision_files[0].read_text(
#                 encoding="utf-8"
#             )
#         )
#
# except Exception as exc:
#     print(
#         "\nPipeline completed, but decision_output "
#         "could not be downloaded from retraining child."
#     )
#     print(
#         "No model will be registered automatically."
#     )
#     print("Download error:", repr(exc))
#     print("Retraining child job:", retrain_child.name)
#
#     raise SystemExit(0)
#
#
# print("\nRetraining decision")
# print("=" * 70)
# print(json.dumps(decision, indent=2))
#
#
# # -------------------------------------------------------------
# # No retraining required
# # -------------------------------------------------------------
# if decision.get("retrain_skipped"):
#
#     print(
#         "\nRetraining correctly skipped:",
#         decision.get("reason"),
#     )
#
#     print(
#         "Current champion remains:",
#         CHAMPION_MODEL,
#     )
#
#     raise SystemExit(0)
#
#
# # -------------------------------------------------------------
# # Challenger trained but rejected
# # -------------------------------------------------------------
# if not decision.get("promote"):
#
#     print(
#         "\nChallenger rejected."
#     )
#
#     print(
#         "Current champion remains:",
#         CHAMPION_MODEL,
#     )
#
#     raise SystemExit(0)
#
#
# # -------------------------------------------------------------
# # Challenger passed promotion criteria
# # -------------------------------------------------------------
#
# # IMPORTANT:
# # Register from the retraining CHILD job output,
# # not the parent pipeline output.
# model_path = (
#     f"azureml://jobs/{retrain_child.name}"
#     f"/outputs/challenger_model"
# )
#
# print("\nRegistering promoted challenger from:")
# print(model_path)
#
# run_model = Model(
#     path=model_path,
#     type=AssetTypes.CUSTOM_MODEL,
#     name=REGISTERED_MODEL_NAME,
#
#     description=(
#         "Retail churn XGBoost challenger promoted "
#         f"from pipeline {returned_job.name}, "
#         f"retraining child {retrain_child.name}, "
#         f"as of {AS_OF_DATE}."
#     ),
#
#     tags={
#         "promotion": "promoted",
#         "as_of_date": AS_OF_DATE,
#         "previous_champion": CHAMPION_MODEL,
#         "source_pipeline_job": returned_job.name,
#         "source_retrain_job": retrain_child.name,
#     },
# )
#
# registered = ml_client.models.create_or_update(
#     run_model
# )
#
# new_model_ref = (
#     f"azureml:{registered.name}:"
#     f"{registered.version}"
# )
#
# print(
#     "\nPromoted challenger registered successfully"
# )
#
# print(
#     "Previous champion:",
#     CHAMPION_MODEL,
# )
#
# print(
#     "New champion:",
#     new_model_ref,
# )
#
# print(
#     "The NEXT monthly pipeline run will "
#     "automatically resolve this version."
# )