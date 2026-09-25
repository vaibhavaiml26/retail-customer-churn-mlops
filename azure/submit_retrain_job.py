"""Submit the Step 10 Azure ML quarterly retraining/champion-challenger job."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

# ---------------------------------------------------------------------
# Fill these with the same workspace values used by your scoring script.
# ---------------------------------------------------------------------
SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"

COMPUTE_NAME = "ChurnAmlCompute"
ENVIRONMENT = "azureml:Customer_Churn_Workspace:2"
BOOTSTRAP_DATA = "azureml:Retail_Churn_Data_Bootstrap:1"

REPLAY_DATA = "azureml:Retail_Churn_Data_Monthly:1"

# Current production champion entering this retraining cycle.
CHAMPION_MODEL = "azureml:retail-churn-xgboost:1"
REGISTERED_MODEL_NAME = "retail-churn-xgboost"

# After August data arrives, run the retrain as of September 1.
AS_OF_DATE = "2011-09-01"

# Simulation/replay split. Keep explicit so cloud behavior cannot silently
# change if config.py is later returned to the richer 3/2/1 dev split.
N_TRAIN = 2
N_VAL = 1
N_TEST = 1
STEP_MONTHS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRAIN_SCRIPT = PROJECT_ROOT / "azure" / "retrain_azure.py"

if not RETRAIN_SCRIPT.exists():
    raise FileNotFoundError(f"Cannot find retraining script: {RETRAIN_SCRIPT}")

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

bootstrap_asset = ml_client.data.get(
    name="Retail_Churn_Data_Bootstrap",
    version="1",
)

replay_asset = ml_client.data.get(
    name="Retail_Churn_Data_Monthly",
    version="1",
)

print("Bootstrap asset:")
print("  type:", bootstrap_asset.type)
print("  path:", bootstrap_asset.path)

print("Replay asset:")
print("  type:", replay_asset.type)
print("  path:", replay_asset.path)

job = command(
    code=str(PROJECT_ROOT),
    command=(
        "python azure/retrain_azure.py "
        "--bootstrap-data ${{inputs.bootstrap_data}} "
        "--replay-data ${{inputs.replay_data}} "
        "--champion-model ${{inputs.champion_model}} "
        "--champion-model-ref ${{inputs.champion_model_ref}} "
        "--challenger-model-output ${{outputs.challenger_model}} "
        "--decision-output ${{outputs.decision_output}} "
        f"--as-of-date {AS_OF_DATE} "
        f"--n-train {N_TRAIN} --n-val {N_VAL} --n-test {N_TEST} "
        f"--step-months {STEP_MONTHS}"
    ),
    inputs={
        "bootstrap_data": Input(
        type=AssetTypes.URI_FOLDER,
        path=bootstrap_asset.path,
        mode="download",
        ),

        "replay_data": Input(
        type=AssetTypes.URI_FOLDER,
        path=replay_asset.path,
        mode="download",

        ),
        "champion_model": Input(
            type=AssetTypes.CUSTOM_MODEL,
            path=CHAMPION_MODEL,
            mode="download",
        ),
        "champion_model_ref": CHAMPION_MODEL,
        },
    outputs={
        "challenger_model": Output(type=AssetTypes.CUSTOM_MODEL),
        "decision_output": Output(type="uri_folder"),
    },
    environment=ENVIRONMENT,
    compute=COMPUTE_NAME,
    display_name="retail-churn-quarterly-retrain",
    experiment_name="retail-churn",
)

returned_job = ml_client.jobs.create_or_update(job)

print("Job submitted:", returned_job.name)
print("Studio URL:", returned_job.studio_url)
print(
    "Challenger model output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/challenger_model",
)
print(
    "Decision output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/decision_output",
)

# Do not use jobs.stream() here. Your local environment has already shown
# a workspace-storage SAS issue while streaming logs. Status polling avoids
# turning a successful Azure job into a misleading local exception.
terminal_states = {"Completed", "Failed", "Canceled", "NotResponding"}
while True:
    current_job = ml_client.jobs.get(returned_job.name)
    print("Job status:", current_job.status)
    if current_job.status in terminal_states:
        break
    time.sleep(20)

if current_job.status != "Completed":
    raise RuntimeError(
        f"Retraining job ended with status {current_job.status}. "
        f"Check Azure ML Studio: {returned_job.studio_url}"
    )

print("Retraining job completed successfully.")

# Read the promotion decision. Azure SDK downloads named outputs under a
# local named-outputs/ tree, so locate the JSON recursively instead of
# depending on one SDK-specific directory layout.
try:
    with tempfile.TemporaryDirectory(prefix="churn_retrain_") as tmp:
        ml_client.jobs.download(
            name=returned_job.name,
            download_path=tmp,
            output_name="decision_output",
        )
        decision_files = list(Path(tmp).rglob("promotion_decision.json"))
        if len(decision_files) != 1:
            raise RuntimeError(
                f"Expected one promotion_decision.json after download; found {len(decision_files)}"
            )
        decision = json.loads(decision_files[0].read_text(encoding="utf-8"))
except Exception as exc:
    print("\nThe Azure retraining job completed, but the local SDK could not download")
    print("decision_output. This can happen with the same workspace-storage SAS")
    print("authentication issue that affected jobs.stream().")
    print("Local download error:", repr(exc))
    print("\nNo model has been registered automatically, which is the safe behavior.")
    print("Open the job in Azure ML Studio -> Outputs + logs -> decision_output ->")
    print("promotion_decision.json. If promote=true, register the challenger_model")
    print("job output as the next retail-churn-xgboost version.")
    raise SystemExit(0)

print("\nPromotion decision")
print("=" * 70)
print(json.dumps(decision, indent=2))

if decision.get("retrain_skipped"):
    print("Retraining was skipped because no new fully resolved snapshot was available.")
    raise SystemExit(0)

if not decision.get("promote"):
    print("\nChallenger rejected. Existing champion remains:", CHAMPION_MODEL)
    raise SystemExit(0)

# The decision passed. Register the entire challenger output folder so the
# model, threshold, metadata, training baseline, prediction baseline, and
# diagnostics remain one versioned Azure model asset.
model_path = f"azureml://jobs/{returned_job.name}/outputs/challenger_model"
run_model = Model(
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    name=REGISTERED_MODEL_NAME,
    description=(
        f"Retail churn XGBoost challenger promoted from retrain job "
        f"{returned_job.name} as of {AS_OF_DATE}."
    ),
    tags={
        "promotion": "promoted",
        "as_of_date": AS_OF_DATE,
        "previous_champion": CHAMPION_MODEL,
        "source_job": returned_job.name,
    },
)
registered = ml_client.models.create_or_update(run_model)
new_model_ref = f"azureml:{registered.name}:{registered.version}"

print("\nPromoted model registered successfully")
print("Name:", registered.name)
print("Version:", registered.version)
print("New champion reference:", new_model_ref)
print("Previous champion:", CHAMPION_MODEL)
print(
    "For subsequent scoring runs, update MODEL in submit_score_job.py to:",
    new_model_ref,
)
