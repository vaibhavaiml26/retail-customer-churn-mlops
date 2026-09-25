from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"

credential = DefaultAzureCredential()

# Force authentication now so errors appear here, not mysteriously later.
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

asset = ml_client.data.get(
    name="Retail_Churn_Data_Bootstrap",
    version="1"
)

print("Name:", asset.name)
print("Version:", asset.version)
print("Type:", asset.type)
print("Path:", asset.path)