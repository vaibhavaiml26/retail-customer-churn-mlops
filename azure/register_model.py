from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"


# IMPORTANT:
# Use the ID/name of the SUCCESSFUL bootstrap run,
# not one of the earlier failed jobs.
JOB_NAME = "tender_button_6wz92l4k8s"

MODEL_NAME = "retail-churn-xgboost"


ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)


model_path = (
    f"azureml://jobs/{JOB_NAME}/outputs/model_output"
)

model = Model(
    name=MODEL_NAME,
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    description=(
        "Retail customer churn XGBoost model trained from "
        "historical bootstrap data."
    ),
)

registered_model = ml_client.models.create_or_update(model)

print("Model registered successfully")
print("Name:", registered_model.name)
print("Version:", registered_model.version)
print("Path:", registered_model.path)