"""Submit the first real retail-churn bootstrap training job to Azure ML."""
from azure.ai.ml import Input, MLClient, Output, command
from azure.identity import DefaultAzureCredential
from pathlib import Path

# ---- Replace only these values if your Azure names differ. -----------------
SUBSCRIPTION_ID = "8e21da87-5b54-48fc-9833-69e109353966"
RESOURCE_GROUP = "AI_ML"
WORKSPACE_NAME = "Customer_Churn"

COMPUTE_NAME = "ChurnAmlCompute"
ENVIRONMENT = "azureml:Customer_Churn_Workspace:2"
BOOTSTRAP_DATA = "azureml:Retail_Churn_Data_Bootstrap:1"

# ---------------------------------------------------------------------------


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    BOOTSTRAP_SCRIPT = PROJECT_ROOT / "azure" / "bootstrap_azure.py"

    print("Project root:", PROJECT_ROOT)
    print("Bootstrap script:", BOOTSTRAP_SCRIPT)
    print("Bootstrap exists:", BOOTSTRAP_SCRIPT.exists())

    if not BOOTSTRAP_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot find bootstrap script: {BOOTSTRAP_SCRIPT}"
        )

    ml_client = MLClient(
        DefaultAzureCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )



    job = command(
        code=str(PROJECT_ROOT),
        command=(
            "python azure/bootstrap_azure.py "
            "--bootstrap-data ${{inputs.bootstrap_data}} "
            "--model-output ${{outputs.model_output}}"
        ),
        inputs={
            "bootstrap_data": Input(
                type="uri_folder",
                path=BOOTSTRAP_DATA,
                mode="download",
            )
        },
        outputs={
            # A custom-model output is a folder. The training script writes
            # model.joblib + metadata/baselines into this named job output.
            "model_output": Output(type="custom_model")
        },
        environment=ENVIRONMENT,
        compute=COMPUTE_NAME,
        display_name="retail-churn-bootstrap-training",
        experiment_name="retail-churn",
        description=(
            "Initial retail churn champion training from the historical "
            "bootstrap data asset using the validated 2/1/1 temporal split."
        ),
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print("Job submitted:", returned_job.name)
    print("Studio URL:", returned_job.studio_url)
    print(
        "Model output URI:",
        f"azureml://jobs/{returned_job.name}/outputs/model_output",
    )

    # Stream stdout/stderr into this terminal until the Azure job completes.
    ml_client.jobs.stream(returned_job.name)


if __name__ == "__main__":
    main()
