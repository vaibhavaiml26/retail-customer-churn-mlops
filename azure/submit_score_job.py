"""Submit one Azure ML monthly churn scoring command job.

Example first replay run:
    python azure/submit_score_job.py

This configuration scores as of 2011-07-01, which means June 2011 data is
available. For later replay runs, change SNAPSHOT_DATE to 2011-08-01,
2011-09-01, etc. The scoring script automatically uses all replay data before
that cutoff and ignores later rows, preventing future leakage.
"""
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# ---------------------------------------------------------------------
# Workspace settings
# ---------------------------------------------------------------------
SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"



# ---------------------------------------------------------------------
# Azure ML assets created in Steps 1-8
# ---------------------------------------------------------------------
COMPUTE_NAME = "ChurnAmlCompute"
ENVIRONMENT = "azureml:Customer_Churn_Workspace:2"

# Both data assets are folders by design. Bootstrap may contain multiple
# historical CSVs; replay_data may contain multiple future/monthly CSVs.
BOOTSTRAP_DATA = "azureml:Retail_Churn_Data_Bootstrap:1"
REPLAY_DATA = "azureml:Retail_Churn_Data_Monthly:1"

# Pin an explicit model version for reproducible replay. Do not use @latest
# while validating historical scoring, because a later registration should not
# silently change an old replay result.
MODEL = "azureml:retail-churn-xgboost:2"

# First scoring run after June 2011 has arrived. Later replay examples:
#   2011-08-01 -> history through July
#   2011-09-01 -> history through August
#   2011-10-01 -> history through September
#   2011-11-01 -> history through October
#   2011-12-01 -> history through November
SNAPSHOT_DATE = "2011-09-01"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = PROJECT_ROOT / "azure" / "score_azure.py"

if not SCORE_SCRIPT.exists():
    raise FileNotFoundError(f"Cannot find Azure scoring script: {SCORE_SCRIPT}")

print("Project root:", PROJECT_ROOT)
print("Score script:", SCORE_SCRIPT)
print("Snapshot date:", SNAPSHOT_DATE)


ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)


job = command(
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
        "bootstrap_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=BOOTSTRAP_DATA,
            mode="download",
        ),
        "replay_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=REPLAY_DATA,
            mode="download",
        ),
        "model": Input(
            type=AssetTypes.CUSTOM_MODEL,
            path=MODEL,
            mode="download",
        ),
        "model_asset_ref": MODEL,
        "snapshot_date": SNAPSHOT_DATE,
    },
    outputs={
        "predictions_output": Output(
            type=AssetTypes.URI_FOLDER,
            mode="upload",
        )
    },
    environment=ENVIRONMENT,
    compute=COMPUTE_NAME,
    display_name=f"retail-churn-score-{SNAPSHOT_DATE}",
    experiment_name="retail-churn-monthly-scoring",
    description=(
        "Historical monthly batch scoring for the retail churn project using "
        "bootstrap + replay history and a pinned registered model version."
    ),
    tags={
        "stage": "monthly_scoring",
        "snapshot_date": SNAPSHOT_DATE,
        "model_asset": MODEL,
    },
)


returned_job = ml_client.jobs.create_or_update(job)

print("Job submitted:", returned_job.name)
print("Studio URL:", returned_job.studio_url)

print(
    "Predictions output URI:",
    f"azureml://jobs/{returned_job.name}/outputs/predictions_output"
)

import time

terminal_states = {
    "Completed",
    "Failed",
    "Canceled",
    "NotResponding",
}

while True:
    current_job = ml_client.jobs.get(returned_job.name)

    print("Job status:", current_job.status)

    if current_job.status in terminal_states:
        break

    time.sleep(20)

if current_job.status == "Completed":
    print("Scoring job completed successfully.")
else:
    raise RuntimeError(
        f"Scoring job ended with status {current_job.status}. "
        f"Check logs in Azure ML Studio: {returned_job.studio_url}"
    )