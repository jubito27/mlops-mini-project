"""Promote the model registered with the Staging alias to Production.

DagsHub's MLflow registry uses model aliases. Avoid the legacy stage-transition
API (/model-versions/transition-stage), which can return HTTP 403.
"""

import os

import mlflow
from mlflow.exceptions import MlflowException


def promote_model() -> None:
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    # DagsHub uses the PAT as both the MLflow tracking username and password.
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "jubito27")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "mlops-mini-project")
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # Registration assigns the newly registered version the Staging alias.
    registered_model = client.get_registered_model(model_name)
    aliases = registered_model.aliases or {}
    staging_version = aliases.get("Staging")
    if not staging_version:
        raise RuntimeError(
            f"No version of '{model_name}' has the 'Staging' alias. "
            "Ensure the model-registration stage completed successfully."
        )

    current_production_version = aliases.get("Production")

    # Preserve the previous production version before moving the Production alias.
    if current_production_version and current_production_version != staging_version:
        client.set_registered_model_alias(
            name=model_name,
            alias="PreviousProduction",
            version=current_production_version,
        )
        print(
            f"Previous production version {current_production_version} "
            "preserved as alias 'PreviousProduction'"
        )

    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=staging_version,
    )

    print(
        f"Model '{model_name}' version {staging_version} "
        "successfully promoted to Production"
    )


if __name__ == "__main__":
    promote_model()
